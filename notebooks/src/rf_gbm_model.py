# -*- coding: utf-8 -*-
"""
RandomForest и Gradient Boosting на тех же признаках, что и логрег-baseline
(без стадии сделки — см. baseline_model.py, почему).
Сравниваем все три модели на одинаковых фолдах, чтобы сравнение было честным.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix

pd.set_option("display.width", 140)

# ============================================================
# ШАГ 1. Данные и признаки — как в baseline_model.py, без изменений
# ============================================================
df = pd.read_csv("сделки_с_признаками.csv")
new_df = df[df["is_existing_client"] == False].copy()

new_df["target"] = (new_df["Стадия сделки"] == "Сделка завершена с успехом").astype(int)

holding_col = new_df["Холдинг"].fillna("НЕТ").replace({"Отсутствует": "НЕТ"})
new_df["is_holding_client"] = (holding_col != "НЕТ").astype(int)

spin_col = "Файл и ссылка продажи (СПИН) (по колонке Продажа)"
presentation_col = "Файл и ссылка продажи (презентация нов.) (по колонке Продажа)"

new_df["log_potential"] = np.log1p(new_df["Потенциал"])
new_df["log_payroll"] = np.log1p(new_df["ФОТ за год"])

numeric_features = [
    "log_potential", "log_payroll", "Штат сотрудников",
    spin_col, presentation_col, "deal_age_days", "is_holding_client",
]
source_dummies = pd.get_dummies(new_df["Источник"], prefix="source", drop_first=True)

X = pd.concat([new_df[numeric_features].reset_index(drop=True), source_dummies.reset_index(drop=True)], axis=1)
y = new_df["target"].reset_index(drop=True)

# ============================================================
# ШАГ 2. Одни и те же фолды для всех трёх моделей — иначе сравнение нечестное
# ============================================================
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
folds = list(skf.split(X, y))

results = {}
oof_predictions = {}

# --- Модель 1: логрег (для сверки с предыдущим шагом) ---
oof_logreg = np.zeros(len(y))
for train_idx, test_idx in folds:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X.iloc[train_idx])
    X_test = scaler.transform(X.iloc[test_idx])
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(X_train, y.iloc[train_idx])
    oof_logreg[test_idx] = model.predict_proba(X_test)[:, 1]
oof_predictions["Логрег"] = oof_logreg

# --- Модель 2: RandomForest ---
# Деревьям не нужна нормализация признаков (StandardScaler не нужен)
oof_rf = np.zeros(len(y))
for train_idx, test_idx in folds:
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=4,          # неглубокие деревья — данных мало, иначе переобучение
        min_samples_leaf=5,   # не даём дереву "запоминать" единичные наблюдения
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    oof_rf[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]
oof_predictions["Random Forest"] = oof_rf

# --- Модель 3: Gradient Boosting (встроенный в sklearn, без доп. установки) ---
oof_gbm = np.zeros(len(y))
for train_idx, test_idx in folds:
    model = HistGradientBoostingClassifier(
        max_depth=3,
        max_iter=150,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    oof_gbm[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]
oof_predictions["Gradient Boosting"] = oof_gbm

# ============================================================
# ШАГ 3. Сравниваем метрики всех трёх моделей
# ============================================================
heuristic_proba = (new_df["priority_score"].reset_index(drop=True) / 100.0).values

def best_f1_and_threshold(y_true, proba):
    thresholds = np.linspace(0.01, 0.99, 99)
    scores = [f1_score(y_true, proba >= t) for t in thresholds]
    i = int(np.argmax(scores))
    return scores[i], thresholds[i]

rows = []
for name, proba in {**oof_predictions, "Эвристика по стадии": heuristic_proba}.items():
    roc = roc_auc_score(y, proba)
    pr = average_precision_score(y, proba)
    f1, t = best_f1_and_threshold(y, proba)
    rows.append({"Модель": name, "ROC-AUC": round(roc, 3), "PR-AUC": round(pr, 3),
                 "Лучший F1": round(f1, 3), "Порог": round(t, 2)})

comparison = pd.DataFrame(rows).set_index("Модель")
print("=" * 70)
print("СРАВНЕНИЕ МОДЕЛЕЙ (все на одинаковых фолдах, без стадии в признаках)")
print("=" * 70)
print(comparison.to_string())

# ============================================================
# ШАГ 4. Кого именно из 7 успешных сделок каждая модель нашла (recall на уровне сделок)
# ============================================================
print("\n" + "=" * 70)
print("КАКИЕ ИЗ 7 УСПЕШНЫХ СДЕЛОК КАЖДАЯ МОДЕЛЬ ПОСТАВИЛА В ТОП-30 ПО ВЕРОЯТНОСТИ")
print("=" * 70)
positives_idx = y[y == 1].index
for name, proba in oof_predictions.items():
    rank = pd.Series(proba).rank(ascending=False)
    found_in_top30 = (rank.loc[positives_idx] <= 30).sum()
    print(f"{name}: {found_in_top30} из 7 успешных сделок попали в топ-30 по вероятности")

# ============================================================
# ШАГ 5. Важность признаков у Random Forest (какие характеристики клиента важнее всего)
# ============================================================
final_rf = RandomForestClassifier(
    n_estimators=300, max_depth=4, min_samples_leaf=5,
    class_weight="balanced", random_state=42, n_jobs=-1,
)
final_rf.fit(X, y)
importances = pd.Series(final_rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n" + "=" * 70)
print("ВАЖНОСТЬ ПРИЗНАКОВ (Random Forest, обучен на всех 574 сделках)")
print("=" * 70)
print(importances.head(10).to_string())

# ============================================================
# ШАГ 6. Сохраняем всё для отчёта
# ============================================================
comparison.to_csv("сравнение_моделей.csv")
new_df["logreg_proba"] = oof_predictions["Логрег"]
new_df["rf_proba"] = oof_predictions["Random Forest"]
new_df["gbm_proba"] = oof_predictions["Gradient Boosting"]
new_df.to_csv("сделки_с_предсказаниями_всех_моделей.csv", index=False)
print("\nСохранены файлы: сравнение_моделей.csv, сделки_с_предсказаниями_всех_моделей.csv")
