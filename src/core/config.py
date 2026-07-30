
from pathlib import Path
from typing import Dict, Union

import yaml
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "src/configs"


class DataConfig(BaseModel):
    raw_data_dir: Path
    filtered_data_dir: Path
    image_dir: str
    classes: dict[int, str]
    filtered_data_path: Path


class ModelConfig(BaseModel):
    model_name: str
    pretrained: bool = True
    image_size: int
    model_path: Path 
    allowed_content_types: set[str] = {"image/jpeg", "image/png", "image/jpg", "image/webp"}
    num_classes: int
    max_image_size_mb: int = 5  # Maximum allowed image size in MB
    confidence_threshold: float  # Confidence threshold for predictions
    iou_threshold: float  # Intersection over Union threshold for predictions



class TrainingConfig(BaseModel):
    output_dir: Path
    run_name: str
    epochs: int
    batch_size: int
    image_size: int
    device: Union[int, str]
    workers: int = 4
    patience: int = 50
    seed: int = 42
    optimizer: str = "auto"
    learning_rate: float = 0.01
    weight_decay: float = 0.0005
    model_name: str 
    pretrained: bool 


class Settings(BaseModel):
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_settings() -> Settings:
    data_config = load_yaml(
        CONFIG_DIR / "data.yaml"
    )

    model_config = load_yaml(
        CONFIG_DIR / "model.yaml"
    )

    training_config = load_yaml(
        CONFIG_DIR / "training.yaml"
    )

    return Settings(
        data=DataConfig(**data_config),
        model=ModelConfig(**model_config),
        training=TrainingConfig(**training_config),
    )


settings = load_settings()

