# -*- coding: utf-8 -*-
"""
Baseline-модель (логрег) для прогноза "закроется ли сделка успехом" —
только на 574 новых клиентах, как решено в дельте.

ВАЖНЫЙ МЕТОДОЛОГИЧЕСКИЙ МОМЕНТ (объяснён подробно в чате):
Стадия сделки НЕ используется как признак модели, потому что она
ПОЧТИ РАВНА target (target=1 <=> Стадия == "Сделка завершена с успехом").
Если дать модели стадию — она не научится ничему полезному, а просто
"подсмотрит" ответ. Поэтому baseline-модель намеренно использует ТОЛЬКО
признаки клиента и сделки, которые не зависят от текущей стадии:
потенциал, ФОТ, штат, СПИН/презентация, источник, холдинг, возраст сделки.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix, classification_report

pd.set_option("display.width", 140)

# ============================================================
# ШАГ 1. Берём только новых клиентов (574) — действующие не участвуют в модели
# ============================================================
df = pd.read_csv("сделки_с_признаками.csv")
new_df = df[df["is_existing_client"] == False].copy()
print("Строк для модели (только новые):", len(new_df))
assert len(new_df) == 574, "Ожидали ровно 574 новых клиента — сегментация разошлась с дельтой!"

# ============================================================
# ШАГ 2. Целевая переменная
# ============================================================
new_df["target"] = (new_df["Стадия сделки"] == "Сделка завершена с успехом").astype(int)
print("Позитивов (успех):", new_df["target"].sum(), "из", len(new_df))

# ============================================================
# ШАГ 3. Признаки — СОЗНАТЕЛЬНО без стадии и без всего, что из неё выведено
# ============================================================
# Холдинг: объединяем три обозначения "пусто" в одно, остальное — как есть
holding_col = new_df["Холдинг"].fillna("НЕТ").replace({"Отсутствует": "НЕТ"})
new_df["is_holding_client"] = (holding_col != "НЕТ").astype(int)

spin_col = "Файл и ссылка продажи (СПИН) (по колонке Продажа)"
presentation_col = "Файл и ссылка продажи (презентация нов.) (по колонке Продажа)"

# Потенциал и ФОТ сильно скошены (единицы огромных значений) — сжимаем логарифмом,
# чтобы модель не "ослепла" от редких выбросов
new_df["log_potential"] = np.log1p(new_df["Потенциал"])
new_df["log_payroll"] = np.log1p(new_df["ФОТ за год"])

numeric_features = [
    "log_potential", "log_payroll", "Штат сотрудников",
    spin_col, presentation_col, "deal_age_days", "is_holding_client",
]

# Источник — категориальный признак с 12 значениями, кодируем one-hot
source_dummies = pd.get_dummies(new_df["Источник"], prefix="source", drop_first=True)

X = pd.concat([new_df[numeric_features].reset_index(drop=True), source_dummies.reset_index(drop=True)], axis=1)
y = new_df["target"].reset_index(drop=True)

print("\nПризнаки модели (", X.shape[1], "шт.):", list(X.columns))

# ============================================================
# ШАГ 4. StratifiedKFold — 5 фолдов (не 10), чтобы в каждом был хотя бы 1 успех
# ============================================================
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_proba = np.zeros(len(y))  # out-of-fold предсказания — честная проверка на "невиданных" данных

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # class_weight='balanced' — компенсируем дисбаланс (7 против 567) весами классов
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    proba = model.predict_proba(X_test_scaled)[:, 1]
    oof_proba[test_idx] = proba

    print(f"Фолд {fold}: {y_test.sum()} позитивов из {len(y_test)}")

# ============================================================
# ШАГ 5. Метрики baseline-модели (честные, по всем out-of-fold предсказаниям)
# ============================================================
roc_auc = roc_auc_score(y, oof_proba)
pr_auc = average_precision_score(y, oof_proba)

# Порог 0.5 для class_weight='balanced' обычно не подходит — подбираем по F1
thresholds = np.linspace(0.01, 0.99, 99)
f1_scores = [f1_score(y, oof_proba >= t) for t in thresholds]
best_t = thresholds[int(np.argmax(f1_scores))]
best_f1 = max(f1_scores)

print("\n" + "=" * 60)
print("МЕТРИКИ BASELINE-МОДЕЛИ (логрег, без стадии как признака)")
print("=" * 60)
print(f"ROC-AUC:              {roc_auc:.3f}")
print(f"PR-AUC (average prec): {pr_auc:.3f}")
print(f"Лучший F1:             {best_f1:.3f} при пороге {best_t:.2f}")

y_pred_best = (oof_proba >= best_t).astype(int)
print("\nМатрица ошибок при этом пороге:")
print(confusion_matrix(y, y_pred_best))
print("\nclassification_report:")
print(classification_report(y, y_pred_best, digits=3))

# ============================================================
# ШАГ 6. Сравнение с эвристикой по стадиям (ВАЖНО: разная информация на входе!)
# ============================================================
heuristic_proba = new_df["priority_score"].reset_index(drop=True) / 100.0
heuristic_roc_auc = roc_auc_score(y, heuristic_proba)
heuristic_pr_auc = average_precision_score(y, heuristic_proba)

print("\n" + "=" * 60)
print("СРАВНЕНИЕ С ЭВРИСТИКОЙ ПО СТАДИЯМ")
print("=" * 60)
print(f"Эвристика (по стадии):  ROC-AUC = {heuristic_roc_auc:.3f}, PR-AUC = {heuristic_pr_auc:.3f}")
print(f"ML baseline (без стадии): ROC-AUC = {roc_auc:.3f}, PR-AUC = {pr_auc:.3f}")
print("""
ВАЖНАЯ ОГОВОРКА для текста ВКР:
эвристика по стадии почти напрямую "видит" ответ (target ОПРЕДЕЛЁН через стадию
"Сделка завершена с успехом"), поэтому её метрики завышены не за счёт качества
скоринга, а потому что использует ту же информацию, что и целевая переменная.
Сравнение честно только в одном смысле: показывает, что баллы, независимые от
стадии (потенциал, ФОТ, источник, штат, СПИН/презентация), несут МЕНЬШЕ прямой
информации об успехе, чем сама стадия — что ожидаемо и логично, а не поражение
модели. Практическая ценность ML baseline — в том, что он может быть применён
К ЛЮБОЙ сделке независимо от её текущей стадии, включая самые ранние, тогда как
эвристика по определению "молчит" почти всё время (5-20%) на ранних стадиях.
""")

# Сохраняем out-of-fold предсказания для дальнейшего анализа (сравнение моделей, отчёт)
new_df["ml_baseline_proba"] = oof_proba
new_df.to_csv("сделки_с_baseline_predictions.csv", index=False)
print("Сохранён файл: сделки_с_baseline_predictions.csv")
