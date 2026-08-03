# -*- coding: utf-8 -*-
"""
Нейросеть с архитектурой 2x8 (два скрытых слоя по 8 нейронов) — по плану из дельты.
Те же признаки, те же фолды, что и в предыдущих моделях — сравнение честное.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

pd.set_option("display.width", 140)

# ============================================================
# ШАГ 1. Данные и признаки — без изменений от предыдущих шагов
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

N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
folds = list(skf.split(X, y))

# ============================================================
# ШАГ 2. Нейросеть 2x8 — важно: нейросети ОБЯЗАТЕЛЬНО нужна нормализация признаков,
# иначе она либо не сойдётся, либо будет учиться очень плохо (в отличие от деревьев)
# ============================================================
# У нейросети нет встроенного class_weight, поэтому вручную считаем веса классов
# и передаём как sample_weight при обучении — компенсируем дисбаланс 7 против 567
n_pos = y.sum()
n_neg = len(y) - n_pos
weight_pos = len(y) / (2 * n_pos)
weight_neg = len(y) / (2 * n_neg)

oof_nn = np.zeros(len(y))
for train_idx, test_idx in folds:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X.iloc[train_idx])
    X_test = scaler.transform(X.iloc[test_idx])
    y_train = y.iloc[train_idx]

    sample_weight = np.where(y_train == 1, weight_pos, weight_neg)

    model = MLPClassifier(
        hidden_layer_sizes=(8, 8),   # архитектура 2x8, как в плане
        activation="relu",
        solver="adam",
        alpha=0.01,                  # L2-регуляризация — данных мало, легко переобучиться
        max_iter=2000,
        random_state=42,
        early_stopping=True,         # остановка, если качество на валидации перестало расти
        validation_fraction=0.15,
    )
    # MLPClassifier.fit не принимает sample_weight напрямую (в отличие от деревьев/логрег) —
    # это архитектурное ограничение sklearn для нейросетей, а не наша недоработка.
    # Обходим через oversampling позитивного класса в обучающей выборке.
    pos_idx_train = np.where(y_train.values == 1)[0]
    neg_idx_train = np.where(y_train.values == 0)[0]
    repeat_factor = int(round(weight_pos / weight_neg))
    oversampled_idx = np.concatenate([neg_idx_train, np.tile(pos_idx_train, repeat_factor)])

    model.fit(X_train[oversampled_idx], y_train.values[oversampled_idx])
    oof_nn[test_idx] = model.predict_proba(X_test)[:, 1]

# ============================================================
# ШАГ 3. Пересчитываем остальные модели на тех же фолдах (для единой таблицы)
# ============================================================
oof_logreg = np.zeros(len(y))
oof_rf = np.zeros(len(y))
oof_gbm = np.zeros(len(y))

for train_idx, test_idx in folds:
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X.iloc[train_idx])
    X_test_s = scaler.transform(X.iloc[test_idx])

    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr.fit(X_train_s, y.iloc[train_idx])
    oof_logreg[test_idx] = lr.predict_proba(X_test_s)[:, 1]

    rf = RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=5,
                                 class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X.iloc[train_idx], y.iloc[train_idx])
    oof_rf[test_idx] = rf.predict_proba(X.iloc[test_idx])[:, 1]

    gbm = HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.05,
                                          class_weight="balanced", random_state=42)
    gbm.fit(X.iloc[train_idx], y.iloc[train_idx])
    oof_gbm[test_idx] = gbm.predict_proba(X.iloc[test_idx])[:, 1]

heuristic_proba = (new_df["priority_score"].reset_index(drop=True) / 100.0).values

all_models = {
    "Логрег": oof_logreg,
    "Random Forest": oof_rf,
    "Gradient Boosting": oof_gbm,
    "Нейросеть 2x8": oof_nn,
    "Эвристика по стадии": heuristic_proba,
}

# ============================================================
# ШАГ 4. Итоговая таблица сравнения всех моделей
# ============================================================
def best_f1_and_threshold(y_true, proba):
    thresholds = np.linspace(0.01, 0.99, 99)
    scores = [f1_score(y_true, proba >= t) for t in thresholds]
    i = int(np.argmax(scores))
    return scores[i], thresholds[i]

rows = []
positives_idx = y[y == 1].index
for name, proba in all_models.items():
    roc = roc_auc_score(y, proba)
    pr = average_precision_score(y, proba)
    f1, t = best_f1_and_threshold(y, proba)
    rank = pd.Series(proba).rank(ascending=False)
    top30 = int((rank.loc[positives_idx] <= 30).sum())
    rows.append({
        "Модель": name, "ROC-AUC": round(roc, 3), "PR-AUC": round(pr, 3),
        "Лучший F1": round(f1, 3), "Порог": round(t, 2), "Успехов в топ-30": f"{top30}/7",
    })

comparison = pd.DataFrame(rows).set_index("Модель")
print("=" * 80)
print("ИТОГОВОЕ СРАВНЕНИЕ ВСЕХ МОДЕЛЕЙ (одинаковые фолды, одинаковые признаки)")
print("=" * 80)
print(comparison.to_string())

comparison.to_csv("итоговое_сравнение_моделей.csv")
new_df["nn_proba"] = oof_nn
new_df["logreg_proba"] = oof_logreg
new_df["rf_proba"] = oof_rf
new_df["gbm_proba"] = oof_gbm
new_df.to_csv("сделки_финальные_предсказания.csv", index=False)
print("\nСохранены файлы: итоговое_сравнение_моделей.csv, сделки_финальные_предсказания.csv")
