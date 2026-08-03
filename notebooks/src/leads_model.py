# -*- coding: utf-8 -*-
"""
Вторая модель: вероятность конверсии лида в сделку.
Тот же принцип, что и для сделок: обучение только на РЕШЁННЫХ лидах
("Нецелевой" = 0, "Переведён в сделку" = 1). "Целевой" и "Необработан" —
ещё не решены, это inference-выборка.
Лиды с источником Холдинг/Платящие/Неплатящие исключены — они относятся
к работе с уже существующими клиентами, а не к привлечению новых
(тот же принцип сегментации, что и для сделок).
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

pd.set_option("display.width", 160)

leads = pd.read_csv("лиды_v2.csv")
leads["created_at"] = pd.to_datetime(leads["created_at"])
leads["status_changed_at"] = pd.to_datetime(leads["status_changed_at"])

EXISTING_SOURCES = {"Холдинг", "Платящие", "Неплатящие"}
leads_new = leads[~leads["source"].isin(EXISTING_SOURCES)].copy()
print("Лидов после исключения источников действующих клиентов:", len(leads_new), "из", len(leads))

leads_new["is_resolved"] = leads_new["lead_status"].isin(["Нецелевой", "Переведён в сделку"])
leads_new["target"] = (leads_new["lead_status"] == "Переведён в сделку").astype(int)

resolved = leads_new[leads_new["is_resolved"]].copy()
unresolved = leads_new[~leads_new["is_resolved"]].copy()
print("Решённых лидов (для обучения):", len(resolved), "| конверсий:", resolved["target"].sum())
print("Нерешённых (inference):", len(unresolved))

# ---- Признаки: источник + давность (сколько дней лид уже находится в системе) ----
snapshot = leads_new["status_changed_at"].max()
for frame in (resolved, unresolved):
    frame["days_since_created"] = (snapshot - frame["created_at"]).dt.days
    frame["days_to_decision"] = (frame["status_changed_at"] - frame["created_at"]).dt.days

combined = pd.concat([resolved, unresolved], ignore_index=True)
source_dummies = pd.get_dummies(combined["source"], prefix="source", drop_first=True)
X_all = pd.concat([combined[["days_since_created"]].reset_index(drop=True),
                    source_dummies.reset_index(drop=True)], axis=1)
X_train = X_all.iloc[:len(resolved)].reset_index(drop=True)
X_infer = X_all.iloc[len(resolved):].reset_index(drop=True)
y = resolved["target"].reset_index(drop=True)

print("Признаков:", X_train.shape[1])

# ============================================================
# Обучение и сравнение (то же трио моделей, что и для сделок)
# ============================================================
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
folds = list(skf.split(X_train, y))

oof = {}
for name, ModelCls, kwargs in [
    ("Логрег", LogisticRegression, dict(class_weight="balanced", max_iter=1000, random_state=42)),
    ("Random Forest", RandomForestClassifier, dict(n_estimators=300, max_depth=4, min_samples_leaf=5,
                                                     class_weight="balanced", random_state=42, n_jobs=-1)),
    ("Gradient Boosting", HistGradientBoostingClassifier, dict(max_depth=3, max_iter=150, learning_rate=0.05,
                                                                  class_weight="balanced", random_state=42)),
]:
    oof_p = np.zeros(len(y))
    for tr, te in folds:
        m = ModelCls(**kwargs)
        m.fit(X_train.iloc[tr], y.iloc[tr])
        oof_p[te] = m.predict_proba(X_train.iloc[te])[:, 1]
    oof[name] = oof_p

rows = []
for name, proba in oof.items():
    roc = roc_auc_score(y, proba)
    pr = average_precision_score(y, proba)
    ts = np.linspace(0.01, 0.99, 99)
    f1s = [f1_score(y, proba >= t) for t in ts]
    rows.append({"Модель": name, "ROC-AUC": round(roc,3), "PR-AUC": round(pr,3), "Лучший F1": round(max(f1s),3)})
comparison = pd.DataFrame(rows).set_index("Модель")
print("\n" + "="*60)
print("СРАВНЕНИЕ МОДЕЛЕЙ КОНВЕРСИИ ЛИД -> СДЕЛКА")
print("="*60)
print(comparison.to_string())
comparison.to_csv("сравнение_моделей_лидов.csv")

# ============================================================
# Важность признаков + инференс на нерешённых лидах
# ============================================================
final_model = RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=5,
                                       class_weight="balanced", random_state=42, n_jobs=-1)
final_model.fit(X_train, y)
importances = pd.Series(final_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print("\nВажность признаков (Random Forest):")
print(importances.head(8).to_string())

unresolved["conversion_proba"] = final_model.predict_proba(X_infer)[:, 1]
top_leads = unresolved.sort_values("conversion_proba", ascending=False).head(15)
print("\nТоп-15 лидов по вероятности конверсии:")
print(top_leads[["id", "lead_status", "source", "days_since_created", "conversion_proba"]].to_string(index=False))

unresolved.to_csv("лиды_с_прогнозом.csv", index=False)
print("\nСохранены файлы: сравнение_моделей_лидов.csv, лиды_с_прогнозом.csv")
