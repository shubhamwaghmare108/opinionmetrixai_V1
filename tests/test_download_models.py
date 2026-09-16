import download_models


def test_models_are_not_ready_when_required_artifacts_are_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(download_models, "MODEL_ROOT", tmp_path)

    assert download_models.models_are_ready() is False


def test_models_are_ready_when_required_artifacts_exist(tmp_path, monkeypatch):
    model_dir = tmp_path / "transformer"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "label_encoder.pkl").write_bytes(b"test")
    monkeypatch.setattr(download_models, "MODEL_ROOT", tmp_path)

    assert download_models.models_are_ready() is True


def test_download_models_skips_download_when_artifacts_exist(tmp_path, monkeypatch):
    model_dir = tmp_path / "transformer"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "label_encoder.pkl").write_bytes(b"test")
    monkeypatch.setattr(download_models, "MODEL_ROOT", tmp_path)

    def fail_if_called(**kwargs):
        raise AssertionError("gdown should not be called for a ready model")

    monkeypatch.setattr(download_models.gdown, "download_folder", fail_if_called)
    download_models.download_models()
