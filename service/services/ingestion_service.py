# -*- coding: utf-8 -*-
"""
Ingestion service: берёт CRM-выгрузку (сейчас — уже очищенный от ПДн CSV,
в проде — сырой Excel), кладёт "как есть" в bronze_deals, затем формирует
Silver-слой: очищенные, дедуплицированные записи + базовые признаки по датам.

Запуск: python -m services.ingestion_service <путь_к_файлу.csv>
"""
import sys
import pandas as pd
import numpy as np
from sqlalchemy import text

sys.path.append("..")
from core.db import engine, init_db, bronze_deals, silver_deals
from core.logging_service import ServiceLogger
from core.feature_logic import STAGE_ORDER

SERVICE_NAME = "ingestion"


def run_ingestion(csv_path: str):
    log = ServiceLogger(engine, SERVICE_NAME)
    init_db()

    with log.timed_step("load_file", f"Читаю файл {csv_path}"):
        df = pd.read_csv(csv_path)
    log.log("load_file_done", f"Загружено строк: {len(df)}", row_count=len(df))

    # ---- Bronze: кладём сырые строки как есть, без изменений ----
    with log.timed_step("write_bronze", "Записываю сырые строки в bronze_deals", row_count=len(df)):
        with engine.begin() as conn:
            for _, row in df.iterrows():
                conn.execute(
                    bronze_deals.insert().values(
                        raw_row=row.where(pd.notnull(row), None).to_dict(),
                        source_file=csv_path,
                    )
                )

    # ---- Дедупликация по Id (на случай повторной загрузки той же выгрузки) ----
    before = len(df)
    df = df.drop_duplicates(subset=["Id"], keep="last")
    after = len(df)
    if before != after:
        log.log("dedup", f"Удалено дублей по Id: {before - after}", row_count=before - after)
    else:
        log.log("dedup", "Дублей по Id не найдено", row_count=0)

    # ---- Даты в нормальный формат ----
    df["Дата создания"] = pd.to_datetime(df["Дата создания"], format="%d.%m.%Y %H:%M")
    df["Дата изменения"] = pd.to_datetime(df["Дата изменения"], format="%d.%m.%Y %H:%M")

    # ---- Объединяем 3 обозначения пустого "Холдинга" в одно ----
    holding_raw = df["Холдинг"]
    holding_clean = holding_raw.fillna("НЕТ").replace({"Отсутствует": "НЕТ"})
    n_merged = int(((holding_raw.isna()) | (holding_raw == "Отсутствует")).sum())
    log.log("merge_holding_empty", f"Объединено обозначений пустого Холдинга: {n_merged}", row_count=n_merged)

    df["holding_clean"] = holding_clean
    df["is_holding_client"] = (holding_clean != "НЕТ")

    # ---- Базовые признаки по датам (не зависят от модели, считаем сразу при загрузке) ----
    snapshot_date = df["Дата изменения"].max()
    df["days_in_pipeline"] = (df["Дата изменения"] - df["Дата создания"]).dt.days
    df["deal_age_days"] = (snapshot_date - df["Дата создания"]).dt.days
    df["stage_number"] = df["Стадия сделки"].map(STAGE_ORDER)

    spin_col = "Файл и ссылка продажи (СПИН) (по колонке Продажа)"
    presentation_col = "Файл и ссылка продажи (презентация нов.) (по колонке Продажа)"

    # ---- Записываем Silver-слой (upsert по deal_id) ----
    with log.timed_step("write_silver", "Записываю очищенные сделки в silver_deals", row_count=len(df)):
        with engine.begin() as conn:
            for _, row in df.iterrows():
                conn.execute(
                    text(
                        """
                        INSERT INTO silver_deals (
                            deal_id, stage, stage_number, is_existing_client, source,
                            holding_name, is_holding_client, potential, payroll_year,
                            staff_count, spin_file_count, presentation_file_count,
                            created_at, changed_at, days_in_pipeline, deal_age_days
                        ) VALUES (
                            :deal_id, :stage, :stage_number, :is_existing_client, :source,
                            :holding_name, :is_holding_client, :potential, :payroll_year,
                            :staff_count, :spin_file_count, :presentation_file_count,
                            :created_at, :changed_at, :days_in_pipeline, :deal_age_days
                        )
                        ON CONFLICT (deal_id) DO UPDATE SET
                            stage=excluded.stage, stage_number=excluded.stage_number,
                            days_in_pipeline=excluded.days_in_pipeline,
                            deal_age_days=excluded.deal_age_days, updated_at=CURRENT_TIMESTAMP
                        """
                    ),
                    {
                        "deal_id": str(row["Id"]),
                        "stage": row["Стадия сделки"],
                        "stage_number": int(row["stage_number"]) if not pd.isna(row["stage_number"]) else None,
                        "is_existing_client": bool(row["is_existing_client"]),
                        "source": row.get("Источник"),
                        "holding_name": row["holding_clean"],
                        "is_holding_client": bool(row["is_holding_client"]),
                        "potential": float(row["Потенциал"]) if not pd.isna(row["Потенциал"]) else None,
                        "payroll_year": float(row["ФОТ за год"]) if not pd.isna(row["ФОТ за год"]) else None,
                        "staff_count": int(row["Штат сотрудников"]) if not pd.isna(row["Штат сотрудников"]) else None,
                        "spin_file_count": int(row[spin_col]) if not pd.isna(row[spin_col]) else 0,
                        "presentation_file_count": int(row[presentation_col]) if not pd.isna(row[presentation_col]) else 0,
                        "created_at": row["Дата создания"].to_pydatetime(),
                        "changed_at": row["Дата изменения"].to_pydatetime(),
                        "days_in_pipeline": int(row["days_in_pipeline"]),
                        "deal_age_days": int(row["deal_age_days"]),
                    },
                )

    log.log("ingestion_complete", f"Ingestion завершён, в Silver {len(df)} сделок", row_count=len(df))
    return len(df)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "сделки_очищено.csv"
    run_ingestion(path)
