from src.transformer_text import light_clean


def test_light_clean_removes_html_and_urls():
    text = "Amazing <b>product</b> https://example.com\nfast delivery"
    assert light_clean(text) == "Amazing product fast delivery"


def test_light_clean_preserves_sentiment_signal():
    text = "I LOVE this!!!"
    assert light_clean(text) == text
