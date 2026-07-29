import logging
import json
from collections import Counter
from ultralytics import YOLO
from src.core.config import settings

logger = logging.getLogger(__name__)


def train_model():
    model = YOLO(
        settings.model.model_name
    )

    model.train(
        data=str(
            settings.data.filtered_data_path
        ),
        epochs=settings.training.epochs,
        batch=settings.training.batch_size,
        imgsz=settings.training.image_size,
        project=str(
            settings.training.output_dir
        ),
        name=settings.training.run_name,
        device=settings.training.device,
        workers=settings.training.workers,
        patience=settings.training.patience,
        seed=settings.training.seed,
        pretrained=settings.model.pretrained,
        optimizer=settings.training.optimizer,
        lr0=settings.training.learning_rate,
        weight_decay=settings.training.weight_decay,
    )

    logger.info(
        "Training completed successfully"
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )

    train_model()
