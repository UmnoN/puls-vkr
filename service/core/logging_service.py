# -*- coding: utf-8 -*-
"""
Единый логгер для всех сервисов. Каждый вызов log_step пишет строку
в таблицу service_logs (видна во всех сервисах и в Streamlit-приложении)
и одновременно печатает в консоль — удобно и для отладки, и для аудита.
"""
import time
import logging
from contextlib import contextmanager
from sqlalchemy import text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
console_logger = logging.getLogger("puls_service")


class ServiceLogger:
    def __init__(self, engine, service_name: str):
        self.engine = engine
        self.service_name = service_name

    def log(self, step: str, message: str, level: str = "INFO",
             row_count: int | None = None, duration_ms: int | None = None):
        console_logger.info(f"[{self.service_name}] {step}: {message}")
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO service_logs (service_name, step, level, message, row_count, duration_ms)
                    VALUES (:service_name, :step, :level, :message, :row_count, :duration_ms)
                    """
                ),
                {
                    "service_name": self.service_name,
                    "step": step,
                    "level": level,
                    "message": message,
                    "row_count": row_count,
                    "duration_ms": duration_ms,
                },
            )

    @contextmanager
    def timed_step(self, step: str, message: str, row_count: int | None = None):
        """Замеряет время шага и пишет лог по завершении — используем через 'with'."""
        start = time.time()
        try:
            yield
            duration_ms = int((time.time() - start) * 1000)
            self.log(step, message, row_count=row_count, duration_ms=duration_ms)
        except Exception as exc:
            duration_ms = int((time.time() - start) * 1000)
            self.log(step, f"ОШИБКА: {exc}", level="ERROR", duration_ms=duration_ms)
            raise
