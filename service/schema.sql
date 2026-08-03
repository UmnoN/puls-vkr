-- ============================================================
-- Схема Postgres для сервиса приоритизации сделок (проект ПУЛЬС / ВКР)
-- Два слоя: Bronze (сырьё "как есть из CRM") и Silver (очищено + признаки + scores)
-- ============================================================

-- ---------- BRONZE: сырые данные, как выгрузили из CRM, ничего не трогаем ----------
CREATE TABLE IF NOT EXISTS bronze_deals (
    id              BIGSERIAL PRIMARY KEY,
    raw_row         JSONB NOT NULL,        -- вся строка Excel как есть, без изменений
    source_file     TEXT NOT NULL,         -- имя файла выгрузки, для трассировки
    loaded_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------- SILVER: очищенные сделки + инженерные признаки (v2) ----------
CREATE TABLE IF NOT EXISTS silver_deals (
    deal_id                     TEXT PRIMARY KEY,
    stage                       TEXT NOT NULL,
    stage_number                SMALLINT,
    is_existing_client          BOOLEAN NOT NULL,
    is_terminal_success         BOOLEAN,
    is_terminal_failure         BOOLEAN,
    is_resolved                 BOOLEAN,        -- v2: завершена ли сделка (успех ИЛИ провал)
    scoring_status              TEXT,           -- v2: active/pending_confirmation/excluded_*
    source                      TEXT,
    industry                    TEXT,           -- v2: отрасль клиента
    holding_name                TEXT,
    is_holding_client           BOOLEAN,
    potential                   NUMERIC,
    payroll_year                NUMERIC,
    staff_count                 INTEGER,
    spin_file_count             SMALLINT,
    presentation_file_count     SMALLINT,
    days_since_last_contact     INTEGER,        -- v2: реальное поле от ПУЛЬС, не приближение
    probability                 NUMERIC,        -- v2: системная вероятность ПУЛЬС
    probability_source          TEXT,
    self_probability             NUMERIC,
    priority_score               NUMERIC,
    priority_category            TEXT,
    weighted_potential           NUMERIC,
    reactivation_score           NUMERIC,
    ml_proba                     NUMERIC,
    is_genuine_existing_success  BOOLEAN,
    updated_at                   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_silver_deals_priority ON silver_deals (priority_score DESC);
CREATE INDEX IF NOT EXISTS idx_silver_deals_stage ON silver_deals (stage);
CREATE INDEX IF NOT EXISTS idx_silver_deals_new ON silver_deals (is_existing_client);
CREATE INDEX IF NOT EXISTS idx_silver_deals_resolved ON silver_deals (is_resolved);

-- ---------- SILVER: лиды (v2, реализовано) ----------
CREATE TABLE IF NOT EXISTS silver_leads (
    lead_id             TEXT PRIMARY KEY,
    source              TEXT,
    lead_status         TEXT,
    is_resolved         BOOLEAN,
    days_since_created  INTEGER,
    conversion_proba    NUMERIC,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------- ЛОГИ: единая таблица для всех сервисов ----------
CREATE TABLE IF NOT EXISTS service_logs (
    id            BIGSERIAL PRIMARY KEY,
    service_name  TEXT NOT NULL,      -- 'ingestion', 'feature_scoring', 'streamlit_app'
    step          TEXT NOT NULL,      -- что за шаг ('load_file', 'dedup', 'train_model', ...)
    level         TEXT NOT NULL DEFAULT 'INFO',  -- INFO / WARNING / ERROR
    message       TEXT NOT NULL,
    row_count     INTEGER,            -- сколько строк затронуто (если применимо)
    duration_ms    INTEGER,            -- сколько шаг занял по времени
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_service_logs_service ON service_logs (service_name, created_at DESC);
