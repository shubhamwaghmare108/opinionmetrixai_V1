import os
import tempfile
from typing import Optional

import pymysql
import streamlit as st

_connection = None
_ssl_cert_path: Optional[str] = None


def get_connection():
    """Create and cache a MySQL connection from Streamlit secrets."""
    global _connection, _ssl_cert_path
    if _connection is not None:
        try:
            _connection.ping(reconnect=True)
            return _connection
        except Exception:
            _connection = None

    if "connections" not in st.secrets or "mysql" not in st.secrets["connections"]:
        raise RuntimeError("MySQL secrets are not configured.")

    mysql_secrets = st.secrets["connections"]["mysql"]
    ssl_dict = None
    cert_content = mysql_secrets.get("ssl_ca")
    if cert_content:
        if _ssl_cert_path is None:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".pem", delete=False) as f:
                f.write(str(cert_content))
                _ssl_cert_path = f.name
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


def save_prediction(review, sentiment, probabilities) -> bool:
    """Persist a prediction; return False instead of breaking the UI."""
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO Prediction
                (review, sentiment, prob_positive, prob_neutral, prob_negative, correct_sentiment)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    review,
                    sentiment,
                    probabilities.get("Positive", 0),
                    probabilities.get("Neutral", 0),
                    probabilities.get("Negative", 0),
                    None,
                ),
            )
        return True
    except Exception as exc:
        st.warning(f"Prediction was generated, but it was not saved to the database: {exc}")
        return False
