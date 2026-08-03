# -*- coding: utf-8 -*-
"""
Streamlit-приложение для менеджера/руководителя:
  - список активных сделок, отсортированный по приоритету
  - отдельно — список кандидатов на реактивацию (провальные сделки)
  - панель логов по сервисам (для прозрачности: кто, что и когда сделал)

Запуск: streamlit run app.py
"""
import sys
import pandas as pd
import streamlit as st

sys.path.append("..")
from core.db import engine

st.set_page_config(page_title="ПУЛЬС — приоритизация сделок", layout="wide")
st.title("Приоритизация сделок — новые клиенты")

TERMINAL_FAILED = "Сделка не состоялась"


@st.cache_data(ttl=60)
def load_deals():
    return pd.read_sql("SELECT * FROM silver_deals WHERE is_existing_client = false", engine)


@st.cache_data(ttl=60)
def load_leads():
    return pd.read_sql("SELECT * FROM silver_leads", engine)


@st.cache_data(ttl=60)
def load_logs(limit=200):
    return pd.read_sql(
        f"SELECT * FROM service_logs ORDER BY created_at DESC LIMIT {limit}", engine
    )


df = load_deals()

if df.empty:
    st.warning("Данных пока нет. Сначала запустите ingestion_service и feature_scoring_service.")
    st.stop()

tab_priority, tab_reactivation, tab_leads, tab_logs = st.tabs(
    ["Приоритет сейчас", "Реактивация провальных", "Лиды", "Логи сервисов"]
)

# ============================================================
# ВКЛАДКА 1 — приоритет активных сделок
# ============================================================
with tab_priority:
    active = df[df["is_resolved"] == 0].copy() if "is_resolved" in df.columns else df[df["stage"] != TERMINAL_FAILED].copy()

    col1, col2, col3 = st.columns(3)
    col1.metric("Активных сделок", len(active))
    col2.metric("Высокий приоритет", int((active["priority_category"] == "высокий").sum()))
    col3.metric("Средний приоритет", int((active["priority_category"] == "средний").sum()))

    category_filter = st.multiselect(
        "Категория приоритета", options=["высокий", "средний", "низкий"],
        default=["высокий", "средний"],
    )
    shown = active[active["priority_category"].isin(category_filter)].sort_values(
        "priority_score", ascending=False
    )

    st.dataframe(
        shown[["deal_id", "stage", "source", "potential", "priority_score",
               "priority_category", "weighted_potential", "ml_proba"]],
        use_container_width=True,
        column_config={
            "priority_score": st.column_config.ProgressColumn("Приоритет", min_value=0, max_value=100),
            "ml_proba": st.column_config.ProgressColumn("ML-вероятность", min_value=0, max_value=1),
        },
    )

# ============================================================
# ВКЛАДКА 2 — реактивация провальных сделок
# ============================================================
with tab_reactivation:
    st.markdown(
        "Провальные сделки не выброшены из системы — у клиента может измениться "
        "бюджет через год. Список ниже отсортирован по свежести провала."
    )
    failed = df[df["stage"] == TERMINAL_FAILED].sort_values("reactivation_score", ascending=False)
    st.dataframe(
        failed[["deal_id", "source", "potential", "days_since_last_contact", "reactivation_score"]],
        use_container_width=True,
        column_config={
            "reactivation_score": st.column_config.ProgressColumn(
                "Шанс реактивации", min_value=0, max_value=100
            ),
        },
    )

# ============================================================
# ВКЛАДКА 3 — лиды: приоритизация по вероятности конверсии в сделку
# ============================================================
with tab_leads:
    leads = load_leads()

    st.markdown(
        "Лиды со статусом «Целевой» и «Необработан» ещё не решены — модель "
        "конверсии оценивает, насколько вероятно, что лид станет сделкой."
    )

    pending = leads[leads["is_resolved"] == 0].copy() if "is_resolved" in leads.columns else leads.copy()

    col1, col2 = st.columns(2)
    col1.metric("Лидов ожидает решения", len(pending))
    col2.metric(
        "Средняя вероятность конверсии",
        f"{pending['conversion_proba'].mean():.0%}" if len(pending) and pending["conversion_proba"].notna().any() else "—",
    )

    source_filter = st.multiselect(
        "Источник", options=sorted(pending["source"].dropna().unique().tolist()),
        default=sorted(pending["source"].dropna().unique().tolist()),
    )
    shown_leads = pending[pending["source"].isin(source_filter)].sort_values(
        "conversion_proba", ascending=False
    )
    st.dataframe(
        shown_leads[["lead_id", "source", "lead_status", "days_since_created", "conversion_proba"]],
        use_container_width=True,
        column_config={
            "conversion_proba": st.column_config.ProgressColumn(
                "Вероятность конверсии", min_value=0, max_value=1
            ),
        },
    )

# ============================================================
# ВКЛАДКА 4 — логи сервисов (прозрачность пайплайна)
# ============================================================
with tab_logs:
    logs = load_logs()
    service_filter = st.multiselect(
        "Сервис", options=logs["service_name"].unique().tolist(),
        default=logs["service_name"].unique().tolist(),
    )
    st.dataframe(
        logs[logs["service_name"].isin(service_filter)],
        use_container_width=True,
    )
