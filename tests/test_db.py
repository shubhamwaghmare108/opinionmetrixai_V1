from unittest.mock import MagicMock

import db


def test_save_prediction_returns_true_and_uses_parameterized_values(monkeypatch):
    cursor = MagicMock()
    connection = MagicMock()
    connection.cursor.return_value.__enter__.return_value = cursor
    monkeypatch.setattr(db, "get_connection", lambda: connection)

    result = db.save_prediction(
        "Great product",
        "Positive",
        {"Positive": 0.9, "Neutral": 0.08, "Negative": 0.02},
    )

    assert result is True
    cursor.execute.assert_called_once()
    query, params = cursor.execute.call_args.args
    assert "INSERT INTO Prediction" in query
    assert params == ("Great product", "Positive", 0.9, 0.08, 0.02, None)


def test_save_prediction_returns_false_when_database_fails(monkeypatch):
    monkeypatch.setattr(db, "get_connection", lambda: (_ for _ in ()).throw(RuntimeError("database unavailable")))
    warning = MagicMock()
    monkeypatch.setattr(db.st, "warning", warning)

    result = db.save_prediction("Review", "Neutral", {})

    assert result is False
    warning.assert_called_once()
