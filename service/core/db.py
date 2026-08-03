# -*- coding: utf-8 -*-
"""
Подключение к базе данных. По умолчанию — локальный SQLite-файл (чтобы можно было
разрабатывать и тестировать без поднятия настоящего Postgres). В проде достаточно
задать переменную окружения DATABASE_URL с адресом Postgres — код не меняется,
т.к. SQLAlchemy сам переводит одни и те же типы под нужный диалект БД.

Пример для прода:
    export DATABASE_URL="postgresql+psycopg2://user:password@host:5432/puls"
"""
import os
from sqlalchemy import (
    create_engine, MetaData, Table, Column,
    BigInteger, Integer, SmallInteger, String, Text, Numeric, Boolean,
    DateTime, JSON, func,
)

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///puls_local.db")

engine = create_engine(DATABASE_URL, future=True)
metadata = MetaData()

bronze_deals = Table(
    "bronze_deals", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("raw_row", JSON, nullable=False),
    Column("source_file", String, nullable=False),
    Column("loaded_at", DateTime, server_default=func.now()),
)

silver_deals = Table(
    "silver_deals", metadata,
    Column("deal_id", String, primary_key=True),
    Column("stage", Text, nullable=False),
    Column("stage_number", SmallInteger),
    Column("is_existing_client", Boolean, nullable=False),
    Column("is_terminal_success", Boolean),
    Column("is_terminal_failure", Boolean),
    Column("is_resolved", Boolean),               # NEW v2: завершена ли сделка
    Column("scoring_status", Text),                # NEW v2: active/pending_confirmation/...
    Column("source", Text),
    Column("industry", Text),                      # NEW v2: отрасль клиента
    Column("holding_name", Text),
    Column("is_holding_client", Boolean),
    Column("potential", Numeric),
    Column("payroll_year", Numeric),
    Column("staff_count", Integer),
    Column("spin_file_count", SmallInteger),
    Column("presentation_file_count", SmallInteger),
    Column("days_since_last_contact", Integer),     # NEW v2: реальное поле, не приближение
    Column("probability", Numeric),                 # NEW v2: системная вероятность ПУЛЬС
    Column("probability_source", Text),
    Column("self_probability", Numeric),
    Column("priority_score", Numeric),
    Column("priority_category", Text),
    Column("weighted_potential", Numeric),
    Column("reactivation_score", Numeric),
    Column("ml_proba", Numeric),
    Column("is_genuine_existing_success", Boolean),
    Column("updated_at", DateTime, server_default=func.now()),
)

silver_leads = Table(
    "silver_leads", metadata,
    Column("lead_id", String, primary_key=True),
    Column("source", Text),
    Column("lead_status", Text),
    Column("is_resolved", Boolean),
    Column("days_since_created", Integer),
    Column("conversion_proba", Numeric),
    Column("updated_at", DateTime, server_default=func.now()),
)

service_logs = Table(
    "service_logs", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("service_name", Text, nullable=False),
    Column("step", Text, nullable=False),
    Column("level", Text, nullable=False, default="INFO"),
    Column("message", Text, nullable=False),
    Column("row_count", Integer),
    Column("duration_ms", Integer),
    Column("created_at", DateTime, server_default=func.now()),
)


def init_db():
    """Создаёт таблицы, если их ещё нет. Безопасно вызывать многократно."""
    metadata.create_all(engine)
