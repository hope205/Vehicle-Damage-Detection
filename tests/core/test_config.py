from pathlib import Path

from src.core.config import load_settings, load_yaml, settings


def test_settings_load_expected_classes():
    assert settings.data.classes[0] == "dent"
    assert settings.data.classes[1] == "scratch"
    assert settings.data.classes[2] == "clean"
    assert settings.model.num_classes == 3
    assert settings.model.max_image_size_mb == 5


def test_model_path_points_at_weights_file():
    model_path = Path(settings.model.model_path)
    # Relative path from repo root as configured in model.yaml
    assert model_path.as_posix().endswith("src/app/model/new_carrd.pt")


def test_load_yaml_missing_file_raises(tmp_path):
    missing = tmp_path / "nope.yaml"
    try:
        load_yaml(missing)
        raised = False
    except FileNotFoundError:
        raised = True
    assert raised is True


def test_load_settings_round_trip():
    loaded = load_settings()
    assert loaded.training.epochs == settings.training.epochs
    assert loaded.model.model_name == settings.model.model_name
