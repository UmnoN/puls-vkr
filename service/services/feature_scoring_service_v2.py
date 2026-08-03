# -*- coding: utf-8 -*-
"""
Feature & scoring service v2. Ключевое отличие от v1: модель обучается
ТОЛЬКО на завершённых сделках (is_resolved=True), остальные получают
инференс-прогноз. Лучшая модель по итогам сравнения (глава 3 ВКР) —
Gradient Boosting. Дополнительно считается модель конверсии лид->сделка.

Запуск: python -m services.feature_scoring_service_v2
"""
import sys
import numpy as np
import pandas as pd
from sqlalchemy import text
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

sys.path.append("..")
from core.db import engine, init_db
from core.logging_service import ServiceLogger
from core.feature_logic import STAGE_HEURISTIC_PCT, TERMINAL_FAILED, priority_category

SERVICE_NAME = "feature_scoring_v2"


def run_feature_scoring():
    log = ServiceLogger(engine, SERVICE_NAME)
    init_db()

    with log.timed_step("read_silver", "Читаю silver_deals и silver_leads"):
        deals = pd.read_sql("SELECT * FROM silver_deals", engine)
        leads = pd.read_sql("SELECT * FROM silver_leads", engine)
    log.log("read_done", f"Сделок: {len(deals)}, лидов: {len(leads)}")

    for col in ["is_existing_client", "is_terminal_success", "is_terminal_failure",
                 "is_resolved", "is_holding_client"]:
        deals[col] = deals[col].astype(bool)

    # ---- Эвристика приоритета — только новые ----
    is_new = ~deals["is_existing_client"]
    deals["priority_score"] = np.where(is_new, deals["stage"].map(STAGE_HEURISTIC_PCT), np.nan)
    deals["priority_category"] = deals["priority_score"].apply(
        lambda s: priority_category(s) if pd.notna(s) else None
    )
    deals["weighted_potential"] = np.where(
        is_new, deals["potential"].fillna(0) * (deals["priority_score"] / 100), np.nan
    )

    # ---- Реактивация (окно 365 дней, реальный days_since_last_contact) ----
    is_failed_new = (deals["stage"] == TERMINAL_FAILED) & is_new
    deals["reactivation_score"] = np.nan
    deals.loc[is_failed_new, "reactivation_score"] = (
        np.clip(1 - deals.loc[is_failed_new, "days_since_last_contact"] / 365, 0, 1) * 100
    )

    deals["is_genuine_existing_success"] = deals["is_existing_client"]  # холдинг уже исключён на входе
    log.log("heuristics_done", "Эвристика/реактивация посчитаны")

    # ============================================================
    # МОДЕЛЬ 1: вероятность закрытия сделки — обучение на resolved-only
    # ============================================================
    resolved = deals[is_new & deals["is_resolved"]].copy()
    unresolved = deals[is_new & ~deals["is_resolved"]].copy()
    y = resolved["is_terminal_success"].astype(int).reset_index(drop=True)

    def build_features(frame):
        f = frame.copy()
        f["log_potential"] = np.log1p(f["potential"].fillna(0))
        f["log_payroll"] = np.log1p(f["payroll_year"].fillna(0))
        f["staff_count"] = f["staff_count"].fillna(0)
        f["days_since_last_contact"] = f["days_since_last_contact"].fillna(
            f["days_since_last_contact"].median()
        )
        return f

    train_f = build_features(resolved)
    infer_f = build_features(unresolved)
    combined = pd.concat([train_f, infer_f], ignore_index=True)
    source_dummies = pd.get_dummies(combined["source"], prefix="source", drop_first=True)
    industry_dummies = pd.get_dummies(combined["industry"].fillna("Не указано"), prefix="ind", drop_first=True)
    numeric_cols = ["log_potential", "log_payroll", "staff_count", "presentation_file_count",
                     "spin_file_count", "days_since_last_contact"]
    X_all = pd.concat([combined[numeric_cols].reset_index(drop=True),
                        source_dummies.reset_index(drop=True),
                        industry_dummies.reset_index(drop=True)], axis=1).fillna(0)
    X_train = X_all.iloc[:len(train_f)].reset_index(drop=True)
    X_infer = X_all.iloc[len(train_f):].reset_index(drop=True)

    with log.timed_step("train_deal_model", "Обучаю Gradient Boosting (resolved-only)", row_count=len(X_train)):
        model = HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.05,
                                                  class_weight="balanced", random_state=42)
        model.fit(X_train, y)
    log.log("train_deal_model_done", f"Обучено на {len(X_train)} сделках, позитивов: {int(y.sum())}")

    unresolved["ml_proba"] = model.predict_proba(X_infer)[:, 1]
    resolved["ml_proba"] = model.predict_proba(X_train)[:, 1]  # для полноты записи в БД
    deals_scored = pd.concat([resolved, unresolved], ignore_index=True)

    # ---- Записываем сделки обратно ----
    with log.timed_step("write_deal_scores", "Записываю scores в silver_deals", row_count=len(deals_scored)):
        with engine.begin() as conn:
            for _, row in deals_scored.iterrows():
                conn.execute(
                    text(
                        """
                        UPDATE silver_deals SET
                            priority_score=:priority_score, priority_category=:priority_category,
                            weighted_potential=:weighted_potential, reactivation_score=:reactivation_score,
                            ml_proba=:ml_proba, is_genuine_existing_success=:genuine,
                            updated_at=CURRENT_TIMESTAMP
                        WHERE deal_id=:deal_id
                        """
                    ),
                    {
                        "deal_id": str(row["deal_id"]),
                        "priority_score": float(row["priority_score"]) if pd.notna(row["priority_score"]) else None,
                        "priority_category": row["priority_category"],
                        "weighted_potential": float(row["weighted_potential"]) if pd.notna(row["weighted_potential"]) else None,
                        "reactivation_score": float(row["reactivation_score"]) if pd.notna(row["reactivation_score"]) else None,
                        "ml_proba": float(row["ml_proba"]) if pd.notna(row["ml_proba"]) else None,
                        "genuine": bool(row["is_genuine_existing_success"]),
                    },
                )
    # Существующие клиенты (не в deals_scored) — тоже проставим genuine флаг
    existing = deals[deals["is_existing_client"]].copy()
    with engine.begin() as conn:
        for _, row in existing.iterrows():
            conn.execute(
                text("UPDATE silver_deals SET is_genuine_existing_success=:g WHERE deal_id=:id"),
                {"g": True, "id": str(row["deal_id"])},
            )

    # ============================================================
    # МОДЕЛЬ 2: конверсия лид -> сделка
    # ============================================================
    EXISTING_SOURCES = {"Холдинг", "Платящие", "Неплатящие"}
    leads_new = leads[~leads["source"].isin(EXISTING_SOURCES)].copy()
    lead_resolved = leads_new[leads_new["is_resolved"].astype(bool)].copy()
    lead_unresolved = leads_new[~leads_new["is_resolved"].astype(bool)].copy()

    if len(lead_resolved) > 10:  # защита от пустой/слишком маленькой выборки
        y_leads = (lead_resolved["lead_status"] == "Переведён в сделку").astype(int).reset_index(drop=True)
        lead_source_dummies = pd.concat([lead_resolved, lead_unresolved], ignore_index=True)
        lead_source_dummies = pd.get_dummies(lead_source_dummies["source"], prefix="source", drop_first=True)
        X_leads_all = pd.concat([
            pd.concat([lead_resolved, lead_unresolved], ignore_index=True)[["days_since_created"]].reset_index(drop=True),
            lead_source_dummies.reset_index(drop=True),
        ], axis=1).fillna(0)
        X_leads_train = X_leads_all.iloc[:len(lead_resolved)].reset_index(drop=True)
        X_leads_infer = X_leads_all.iloc[len(lead_resolved):].reset_index(drop=True)

        with log.timed_step("train_lead_model", "Обучаю модель конверсии лид->сделка", row_count=len(X_leads_train)):
            lead_model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
            lead_model.fit(X_leads_train, y_leads)
        log.log("train_lead_model_done", f"Обучено на {len(X_leads_train)} лидах, конверсий: {int(y_leads.sum())}")

        lead_unresolved["conversion_proba"] = lead_model.predict_proba(X_leads_infer)[:, 1]
        lead_resolved["conversion_proba"] = lead_model.predict_proba(X_leads_train)[:, 1]
        leads_scored = pd.concat([lead_resolved, lead_unresolved], ignore_index=True)

        with log.timed_step("write_lead_scores", "Записываю conversion_proba в silver_leads", row_count=len(leads_scored)):
            with engine.begin() as conn:
                for _, row in leads_scored.iterrows():
                    conn.execute(
                        text("UPDATE silver_leads SET conversion_proba=:p, updated_at=CURRENT_TIMESTAMP WHERE lead_id=:id"),
                        {"p": float(row["conversion_proba"]), "id": str(row["lead_id"])},
                    )
    else:
        log.log("skip_lead_model", "Недостаточно решённых лидов для обучения", level="WARNING")

    log.log("scoring_v2_complete", "Feature & scoring v2 завершён")


if __name__ == "__main__":
    run_feature_scoring()
