# -*- coding: utf-8 -*-
"""
Ingestion service v2 — работает с более полной выгрузкой ПУЛЬС
(bauman_dataset), включающей явные флаги is_terminal_success/failure,
отрасль, реальное число дней без контакта и лист лидов.
Персональные/идентифицирующие поля (company_name, inn, manager, comment)
должны быть удалены ДО подачи файла в этот сервис (см. README).

Запуск: python -m services.ingestion_service_v2 <сделки.csv> <лиды.csv>
"""
import sys
import pandas as pd
from sqlalchemy import text

sys.path.append("..")
from core.db import engine, init_db
from core.logging_service import ServiceLogger
from core.feature_logic import STAGE_ORDER

SERVICE_NAME = "ingestion_v2"


def run_ingestion(deals_csv: str, leads_csv: str):
    log = ServiceLogger(engine, SERVICE_NAME)
    init_db()

    with log.timed_step("load_deals", f"Читаю {deals_csv}"):
        deals = pd.read_csv(deals_csv)
    log.log("load_deals_done", f"Сделок загружено: {len(deals)}", row_count=len(deals))

    with log.timed_step("load_leads", f"Читаю {leads_csv}"):
        leads = pd.read_csv(leads_csv)
    log.log("load_leads_done", f"Лидов загружено: {len(leads)}", row_count=len(leads))

    # Объединяем 3 обозначения пустого холдинга (повторяющаяся особенность выгрузки)
    holding_clean = deals["holding_name"].fillna("НЕТ").replace({"Отсутствует": "НЕТ"})
    deals["is_holding_client"] = holding_clean != "НЕТ"
    deals["is_existing_client"] = deals["client_cohort"] == "existing"
    deals["is_resolved"] = deals["is_terminal_success"] | deals["is_terminal_failure"]
    deals["stage_number"] = deals["stage"].map(STAGE_ORDER)

    with log.timed_step("write_silver_deals", "Записываю сделки в silver_deals", row_count=len(deals)):
        with engine.begin() as conn:
            for _, row in deals.iterrows():
                conn.execute(
                    text(
                        """
                        INSERT INTO silver_deals (
                            deal_id, stage, stage_number, is_existing_client,
                            is_terminal_success, is_terminal_failure, is_resolved,
                            scoring_status, source, industry, holding_name,
                            is_holding_client, potential, payroll_year, staff_count,
                            spin_file_count, presentation_file_count,
                            days_since_last_contact, probability, probability_source,
                            self_probability
                        ) VALUES (
                            :deal_id, :stage, :stage_number, :is_existing_client,
                            :is_terminal_success, :is_terminal_failure, :is_resolved,
                            :scoring_status, :source, :industry, :holding_name,
                            :is_holding_client, :potential, :payroll_year, :staff_count,
                            :spin_file_count, :presentation_file_count,
                            :days_since_last_contact, :probability, :probability_source,
                            :self_probability
                        )
                        ON CONFLICT (deal_id) DO UPDATE SET
                            stage=excluded.stage, is_resolved=excluded.is_resolved,
                            scoring_status=excluded.scoring_status,
                            probability=excluded.probability, updated_at=CURRENT_TIMESTAMP
                        """
                    ),
                    {
                        "deal_id": str(row["id"]),
                        "stage": row["stage"],
                        "stage_number": int(row["stage_number"]) if pd.notna(row["stage_number"]) else None,
                        "is_existing_client": bool(row["is_existing_client"]),
                        "is_terminal_success": bool(row["is_terminal_success"]),
                        "is_terminal_failure": bool(row["is_terminal_failure"]),
                        "is_resolved": bool(row["is_resolved"]),
                        "scoring_status": row.get("scoring_status"),
                        "source": row.get("source"),
                        "industry": row.get("industry") if pd.notna(row.get("industry")) else None,
                        "holding_name": row.get("holding_name") if pd.notna(row.get("holding_name")) else None,
                        "is_holding_client": bool(row["is_holding_client"]),
                        "potential": float(row["potential_millions"]) if pd.notna(row["potential_millions"]) else None,
                        "payroll_year": float(row["payroll_millions"]) if pd.notna(row["payroll_millions"]) else None,
                        "staff_count": int(row["staff_count"]) if pd.notna(row["staff_count"]) else None,
                        "spin_file_count": int(bool(row["spin_exists"])),
                        "presentation_file_count": int(bool(row["presentation_exists"])),
                        "days_since_last_contact": int(row["days_since_last_contact"]) if pd.notna(row["days_since_last_contact"]) else None,
                        "probability": float(row["probability"]) if pd.notna(row["probability"]) else None,
                        "probability_source": row.get("probability_source"),
                        "self_probability": float(row["self_probability"]) if pd.notna(row["self_probability"]) else None,
                    },
                )

    # ---- Лиды ----
    leads["created_at"] = pd.to_datetime(leads["created_at"])
    EXISTING_SOURCES = {"Холдинг", "Платящие", "Неплатящие"}
    leads["is_resolved"] = leads["lead_status"].isin(["Нецелевой", "Переведён в сделку"])
    snapshot = leads["status_changed_at"].apply(pd.Timestamp).max() if "status_changed_at" in leads else pd.Timestamp.now()
    leads["days_since_created"] = (pd.Timestamp(snapshot) - leads["created_at"]).dt.days

    with log.timed_step("write_silver_leads", "Записываю лиды в silver_leads", row_count=len(leads)):
        with engine.begin() as conn:
            for _, row in leads.iterrows():
                conn.execute(
                    text(
                        """
                        INSERT INTO silver_leads (lead_id, source, lead_status, is_resolved, days_since_created)
                        VALUES (:lead_id, :source, :lead_status, :is_resolved, :days_since_created)
                        ON CONFLICT (lead_id) DO UPDATE SET
                            lead_status=excluded.lead_status, is_resolved=excluded.is_resolved,
                            updated_at=CURRENT_TIMESTAMP
                        """
                    ),
                    {
                        "lead_id": str(row["id"]),
                        "source": row.get("source"),
                        "lead_status": row["lead_status"],
                        "is_resolved": bool(row["is_resolved"]),
                        "days_since_created": int(row["days_since_created"]) if pd.notna(row["days_since_created"]) else None,
                    },
                )

    log.log("ingestion_v2_complete", f"Загружено {len(deals)} сделок и {len(leads)} лидов",
             row_count=len(deals) + len(leads))


if __name__ == "__main__":
    deals_path = sys.argv[1] if len(sys.argv) > 1 else "сделки_v2.csv"
    leads_path = sys.argv[2] if len(sys.argv) > 2 else "лиды_v2.csv"
    run_ingestion(deals_path, leads_path)
