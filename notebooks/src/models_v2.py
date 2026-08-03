# -*- coding: utf-8 -*-
"""
Модели на данных v2. Обучение ТОЛЬКО на завершённых новых сделках (274:
7 успехов + 267 провалов). Активные (297) получают инференс-прогноз, но
не участвуют в обучении и оценке метрик.
Новое по сравнению с версией 1: отрасль (industry) как признак, реальный
days_since_last_contact вместо приближения, дополнительное сравнение с
системной вероятностью ПУЛЬС (probability, поле "уровня 2/3", не участвует
как признак — используется только для сравнения на графике метрик).
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

pd.set_option("display.width", 160)

df = pd.read_csv("сделки_с_признаками_v2.csv")

new_resolved = df[(~df["is_existing_client"]) & (df["is_resolved"])].copy()
new_unresolved = df[(~df["is_existing_client"]) & (~df["is_resolved"])].copy()

y = new_resolved["target"].reset_index(drop=True)

# ============================================================
# Признаки — по-прежнему БЕЗ стадии (та же причина: стадия почти равна target)
# Новое: industry (one-hot), реальный days_since_last_contact
# ============================================================
def build_features(frame):
    f = frame.copy()
    f["log_potential"] = np.log1p(f["potential_millions"].fillna(0))
    f["log_payroll"] = np.log1p(f["payroll_millions"].fillna(0))
    f["staff_count"] = f["staff_count"].fillna(0)
    holding_clean = f["holding_name"].fillna("НЕТ").replace({"Отсутствует": "НЕТ"})
    f["is_holding_client"] = (holding_clean != "НЕТ").astype(int)
    f["presentation_exists"] = f["presentation_exists"].astype(int)
    f["spin_exists"] = f["spin_exists"].astype(int)
    f["days_since_last_contact"] = f["days_since_last_contact"].fillna(f["days_since_last_contact"].median())
    return f

train_f = build_features(new_resolved)
infer_f = build_features(new_unresolved)

numeric_cols = ["log_potential", "log_payroll", "staff_count", "presentation_exists",
                 "spin_exists", "days_since_last_contact", "is_holding_client"]

# Категориальные: source + industry, one-hot по ОБЪЕДИНЁННОМУ набору категорий
# (train + infer вместе), чтобы столбцы совпадали при инференсе
combined = pd.concat([train_f, infer_f], axis=0, ignore_index=True)
source_dummies = pd.get_dummies(combined["source"], prefix="source", drop_first=True)
industry_dummies = pd.get_dummies(combined["industry"].fillna("Не указано"), prefix="ind", drop_first=True)

X_combined = pd.concat([combined[numeric_cols].reset_index(drop=True),
                         source_dummies.reset_index(drop=True),
                         industry_dummies.reset_index(drop=True)], axis=1)
X_train = X_combined.iloc[:len(train_f)].reset_index(drop=True)
X_infer = X_combined.iloc[len(train_f):].reset_index(drop=True)

print("Признаков всего:", X_train.shape[1], "| Обучающих строк:", len(X_train), "| Инференс-строк:", len(X_infer))

# ============================================================
# Кросс-валидация и обучение трёх моделей (без нейросети в этом проходе —
# уже показали её ограничения на версии 1, при ещё меньшем resolved-наборе
# результат будет хуже; можно добавить по запросу)
# ============================================================
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
folds = list(skf.split(X_train, y))

oof = {}

oof_lr = np.zeros(len(y))
for tr, te in folds:
    m = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    m.fit(X_train.iloc[tr], y.iloc[tr])
    oof_lr[te] = m.predict_proba(X_train.iloc[te])[:, 1]
oof["Логрег"] = oof_lr

oof_rf = np.zeros(len(y))
for tr, te in folds:
    m = RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=5,
                                 class_weight="balanced", random_state=42, n_jobs=-1)
    m.fit(X_train.iloc[tr], y.iloc[tr])
    oof_rf[te] = m.predict_proba(X_train.iloc[te])[:, 1]
oof["Random Forest"] = oof_rf

oof_gbm = np.zeros(len(y))
for tr, te in folds:
    m = HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.05,
                                         class_weight="balanced", random_state=42)
    m.fit(X_train.iloc[tr], y.iloc[tr])
    oof_gbm[te] = m.predict_proba(X_train.iloc[te])[:, 1]
oof["Gradient Boosting"] = oof_gbm

# Наша эвристика и системная вероятность ПУЛЬС — для сравнения (не признаки!)
heuristic_proba = (new_resolved["priority_score"].values / 100)
puls_probability = (new_resolved["probability"].fillna(0).values / 100)

def best_f1(y_true, proba):
    ts = np.linspace(0.01, 0.99, 99)
    scores = [f1_score(y_true, proba >= t) for t in ts]
    i = int(np.argmax(scores))
    return scores[i], ts[i]

print("\n" + "=" * 75)
print("СРАВНЕНИЕ (обучение и оценка только на 274 завершённых новых сделках)")
print("=" * 75)
rows = []
positives_idx = y[y == 1].index
for name, proba in {**oof, "Наша эвристика (стадия)": heuristic_proba,
                     "Системная probability ПУЛЬС": puls_probability}.items():
    roc = roc_auc_score(y, proba)
    pr = average_precision_score(y, proba)
    f1, t = best_f1(y, proba)
    rank = pd.Series(proba).rank(ascending=False)
    top30 = int((rank.loc[positives_idx] <= 30).sum())
    rows.append({"Модель": name, "ROC-AUC": round(roc,3), "PR-AUC": round(pr,3),
                 "Лучший F1": round(f1,3), "Успехов в топ-30": f"{top30}/7"})
comparison = pd.DataFrame(rows).set_index("Модель")
print(comparison.to_string())
comparison.to_csv("сравнение_моделей_v2.csv")

# ============================================================
# Важность признаков Random Forest (на полном resolved-наборе)
# ============================================================
final_rf = RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=5,
                                    class_weight="balanced", random_state=42, n_jobs=-1)
final_rf.fit(X_train, y)
importances = pd.Series(final_rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print("\nТоп-10 важных признаков (Random Forest):")
print(importances.head(10).to_string())

# ============================================================
# ИНФЕРЕНС: прогноз для 297 активных сделок — практический результат
# ============================================================
new_unresolved["ml_proba"] = final_rf.predict_proba(X_infer)[:, 1]
top_active = new_unresolved.sort_values("ml_proba", ascending=False).head(15)
print("\nТоп-15 активных сделок по прогнозной вероятности успеха (ML):")
print(top_active[["id", "stage", "potential_millions", "ml_proba"]].to_string(index=False))

new_unresolved.to_csv("активные_сделки_с_прогнозом.csv", index=False)
print("\nСохранены файлы: сравнение_моделей_v2.csv, активные_сделки_с_прогнозом.csv")
