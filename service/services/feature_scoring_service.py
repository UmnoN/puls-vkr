# -*- coding: utf-8 -*-
"""
Feature & scoring service: читает Silver-слой, считает:
  - priority_score / priority_category (эвристика по стадии, пороги 40/70 из ПУЛЬС)
  - weighted_potential (потенциал * вероятность, метрика director_board из ПУЛЬС)
  - reactivation_score (для "Сделка не состоялась" — давность провала, окно 365 дней)
  - ml_proba (Random Forest, обучен на 574 новых клиентах, БЕЗ стадии как признака —
    см. объяснение про утечку целевой переменной через стадию, обсуждали в чате)

Запуск: python -m services.feature_scoring_service
"""
import sys
import numpy as np
import pandas as pd
from sqlalchemy import text
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

sys.path.append("..")
from core.db import engine, init_db
from core.logging_service import ServiceLogger
from core.feature_logic import (
    STAGE_HEURISTIC_PCT, TERMINAL_FAILED, TERMINAL_SUCCESS, priority_category,
)

SERVICE_NAME = "feature_scoring"


def run_feature_scoring():
    log = ServiceLogger(engine, SERVICE_NAME)
    init_db()

    with log.timed_step("read_silver", "Читаю silver_deals"):
        df = pd.read_sql("SELECT * FROM silver_deals", engine)
    log.log("read_silver_done", f"Прочитано сделок: {len(df)}", row_count=len(df))

    if df.empty:
        log.log("no_data", "silver_deals пуст — сначала запустите ingestion_service", level="WARNING")
        return

    df["changed_at"] = pd.to_datetime(df["changed_at"])
    snapshot_date = df["changed_at"].max()
    # SQLite хранит boolean как 0/1 (int) — приводим явно один раз, в самом начале,
    # иначе ~ на int сработает как побитовое НЕ (0 -> -1), а не логическое отрицание
    df["is_existing_client"] = df["is_existing_client"].astype(bool)
    df["is_holding_client"] = df["is_holding_client"].astype(bool)

    # ---- Эвристика приоритета по стадии — ТОЛЬКО для новых клиентов ----
    # Уточнение владельца проекта: у действующих клиентов "Стадия" — служебная метка
    # вовлечённости менеджера (провалена = "не веду активно сейчас"), а НЕ статус
    # успеха сделки. CRM кастомная, отдельную воронку для действующих клиентов
    # сделать пока нельзя (нет ресурсов разработки) — поэтому НЕ считаем эвристику
    # по стадии для действующих, иначе это будет вводящей в заблуждение метрикой.
    is_new = ~df["is_existing_client"]
    df["priority_score"] = np.nan
    df.loc[is_new, "priority_score"] = df.loc[is_new, "stage"].map(STAGE_HEURISTIC_PCT)
    df["priority_category"] = df["priority_score"].apply(
        lambda s: priority_category(s) if pd.notna(s) else None
    )
    df["weighted_potential"] = np.nan
    df.loc[is_new, "weighted_potential"] = (
        df.loc[is_new, "potential"].fillna(0) * (df.loc[is_new, "priority_score"] / 100)
    )
    log.log("priority_scored", "Посчитан priority_score / weighted_potential (только новые)", row_count=int(is_new.sum()))

    # ---- Реактивация провальных сделок — тоже только новые (окно 365 дней) ----
    is_failed = (df["stage"] == TERMINAL_FAILED) & is_new
    days_since_change = (snapshot_date - df["changed_at"]).dt.days
    df["reactivation_score"] = np.nan
    df.loc[is_failed, "reactivation_score"] = (
        np.clip(1 - days_since_change[is_failed] / 365, 0, 1) * 100
    )
    log.log("reactivation_scored", f"Посчитан reactivation_score для {int(is_failed.sum())} провальных новых сделок",
             row_count=int(is_failed.sum()))

    # ---- "Реальный успех" у действующих клиентов (кроме источника Холдинг) ----
    df["is_genuine_existing_success"] = df["is_existing_client"] & (df["source"] != "Холдинг")

    # ---- ML-модель: только новые клиенты, БЕЗ стадии в признаках ----
    # SQLite хранит boolean как 0/1 (int), поэтому приводим явно, иначе ~ сработает
    # как побитовое НЕ над числом (0 -> -1), а не как логическое отрицание
    new_df = df[~df["is_existing_client"]].copy()
    new_df["target"] = (new_df["stage"] == TERMINAL_SUCCESS).astype(int)

    numeric_features = ["potential", "payroll_year", "staff_count",
                         "spin_file_count", "presentation_file_count",
                         "deal_age_days", "is_holding_client"]
    for c in numeric_features:
        new_df[c] = pd.to_numeric(new_df[c], errors="coerce").fillna(0)
    new_df["log_potential"] = np.log1p(new_df["potential"])
    new_df["log_payroll"] = np.log1p(new_df["payroll_year"])

    source_dummies = pd.get_dummies(new_df["source"], prefix="source", drop_first=True)
    feature_cols = ["log_potential", "log_payroll", "staff_count",
                     "spin_file_count", "presentation_file_count",
                     "deal_age_days", "is_holding_client"]
    X = pd.concat([new_df[feature_cols].reset_index(drop=True), source_dummies.reset_index(drop=True)], axis=1)
    y = new_df["target"].reset_index(drop=True)

    with log.timed_step("train_model", "Обучаю Random Forest (StratifiedKFold, 5 фолдов)", row_count=len(X)):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof_proba = np.zeros(len(y))
        for train_idx, test_idx in skf.split(X, y):
            model = RandomForestClassifier(
                n_estimators=300, max_depth=4, min_samples_leaf=5,
                class_weight="balanced", random_state=42, n_jobs=-1,
            )
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            oof_proba[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]
    log.log("train_model_done", f"ML-модель обучена, позитивов: {int(y.sum())} из {len(y)}", row_count=len(y))

    new_df["ml_proba"] = oof_proba
    df = df.drop(columns=["ml_proba"]).merge(new_df[["deal_id", "ml_proba"]], on="deal_id", how="left")

    # ---- Записываем результат обратно в silver_deals ----
    with log.timed_step("write_scores", "Записываю scores обратно в silver_deals", row_count=len(df)):
        with engine.begin() as conn:
            for _, row in df.iterrows():
                conn.execute(
                    text(
                        """
                        UPDATE silver_deals SET
                            priority_score = :priority_score,
                            priority_category = :priority_category,
                            weighted_potential = :weighted_potential,
                            reactivation_score = :reactivation_score,
                            ml_proba = :ml_proba,
                            is_genuine_existing_success = :is_genuine_existing_success,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE deal_id = :deal_id
                        """
                    ),
                    {
                        "deal_id": str(row["deal_id"]),
                        "priority_score": float(row["priority_score"]) if not pd.isna(row["priority_score"]) else None,
                        "priority_category": row["priority_category"],
                        "weighted_potential": float(row["weighted_potential"]) if not pd.isna(row["weighted_potential"]) else None,
                        "reactivation_score": float(row["reactivation_score"]) if not pd.isna(row["reactivation_score"]) else None,
                        "ml_proba": float(row["ml_proba"]) if not pd.isna(row["ml_proba"]) else None,
                        "is_genuine_existing_success": bool(row["is_genuine_existing_success"]),
                    },
                )

    log.log("scoring_complete", "Feature & scoring завершён", row_count=len(df))
    return df


if __name__ == "__main__":
    run_feature_scoring()
