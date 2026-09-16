"""Optional MySQL persistence for prediction results."""

from __future__ import annotations

import logging
import tempfile
from typing import Any

import pymysql
import streamlit as st

logger = logging.getLogger(__name__)
_connection = None
_ssl_cert_path: str | None = None


def get_connection():
    """Create and cache a MySQL connection from Streamlit secrets."""
    global _connection, _ssl_cert_path
    if _connection is not None:
        try:
            _connection.ping(reconnect=True)
            return _connection
        except Exception:
            _connection = None

    connections = st.secrets.get("connections", {})
    mysql_secrets = connections.get("mysql") if hasattr(connections, "get") else None
    if not mysql_secrets:
        raise RuntimeError("MySQL secrets are not configured.")

    ssl_dict = None
    cert_content = mysql_secrets.get("ssl_ca")
    if cert_content:
        if _ssl_cert_path is None:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".pem", delete=False, encoding="utf-8") as handle:
                handle.write(str(cert_content))
                _ssl_cert_path = handle.name
        ssl_dict = {"ca": _ssl_cert_path}

    _connection = pymysql.connect(
        host=mysql_secrets["host"],
        user=mysql_secrets["username"],
        password=mysql_secrets["password"],
        database=mysql_secrets["database"],
        port=int(mysql_secrets.get("port", 3306)),
        ssl=ssl_dict,
        autocommit=True,
        connect_timeout=10,
    )
    return _connection


def save_prediction(review: str, sentiment: str, probabilities: dict[str, float]) -> bool:
    """Persist a prediction when MySQL is configured; otherwise return False."""
    try:
        connection = get_connection()
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO Prediction
                (review, sentiment, prob_positive, prob_neutral, prob_negative, correct_sentiment)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    review,
                    sentiment,
                    probabilities.get("Positive", 0.0),
                    probabilities.get("Neutral", 0.0),
                    probabilities.get("Negative", 0.0),
                    None,
                ),
            )
        return True
    except Exception:
        logger.exception("Prediction persistence failed")
        return False
