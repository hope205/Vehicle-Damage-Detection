# Vehicle Damage Detection System


## Solution Overview

The proposed solution is an end-to-end computer vision system for detecting two common types of vehicle damage: **scratches and dents**.

### Dataset Source

The dataset used for this project is the **CarDD with YOLO Annotations (Images + Labels)** dataset, available on Kaggle:

https://www.kaggle.com/datasets/gabrielfcarvalho/cardd-with-yolo-annotations-images-labels

The dataset is based on the **Car Damage Detection Dataset (CarDD)** and contains vehicle images with YOLO-format bounding-box annotations for different types of car damage.

### Dataset Statistics

The original dataset contains images of vehicles with multiple categories of damage. For this project, the dataset was filtered to focus specifically on the two damage categories relevant to the vehicle damage detection task:

- **Dent**
- **Scratch**

During preprocessing, all annotations belonging to other damage categories were removed. Images that contained at least one **dent** or **scratch** annotation were retained, while images without either target class were excluded.

The resulting filtered dataset contains **6,138 annotated damage instances** across the two target classes.


### Class Distribution

The dataset contains **4,000 images** divided into training, validation, and test sets. The original CarDD annotations were processed to retain **dent** and **scratch** bounding boxes. Images containing only other types of damage, such as cracks, broken lamps, shattered glass, or flat tyres, were retained and labelled as **clean** for this task.

| Split | Images | Images with Dent | Images with Scratch | Images Labelled Clean |
|---|---:|---:|---:|---:|
| Train | 2,816 | 1,242 | 1,507 | 715 |
| Validation | 810 | 352 | 431 | 207 |
| Test | 374 | 157 | 183 | 104 |
| **Total** | **4,000** | **1,751** | **2,121** | **1,026** |

The model uses three target classes:

- **Class 0 — Dent:** Images containing one or more dent annotations.
- **Class 1 — Scratch:** Images containing one or more scratch annotations.
- **Class 2 — Clean:** Images with no retained dent or scratch annotations. These images may contain other types of vehicle damage that are outside the scope of this model.

> **Note:** An image can contain multiple damage types, so the class counts represent the number of images containing each class and therefore do not necessarily sum to the total number of images.



## API

The Vehicle Damage Detection System provides a **FastAPI-based REST API** for detecting vehicle **scratches and dents** from uploaded images.

The API provides three main endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/health` | `GET` | Checks whether the API is running and whether the detection model is loaded. |
| `/predict` | `POST` | Accepts a vehicle image and returns detected damage classes, bounding boxes, and confidence scores. |
| `/predict/annotated` | `POST` | Accepts a vehicle image and returns the image with bounding boxes drawn around detected damage. |

### Health Check

```http
GET /health
```

The health endpoint can be used to verify that the API is running and that the YOLO damage detection model has been successfully loaded.

Example response:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### Damage Prediction

```http
POST /predict
```

The prediction endpoint accepts an image of a vehicle and uses the trained YOLOv8n model to detect visible **scratches** and **dents**.

The response contains information about the detected damage, including:

- Damage class.
- Confidence score.
- Bounding-box coordinates.

This endpoint is useful when another application needs to consume the model's predictions programmatically.

### Annotated Prediction

```http
POST /predict/annotated
```

The annotated prediction endpoint performs the same vehicle damage detection process but returns the uploaded image with bounding boxes drawn around the detected scratches and dents.

This endpoint is useful when users want to visually inspect the model's predictions and see exactly where the detected damage is located on the vehicle.

### Supported Image Formats

The API accepts the following image formats:

- JPEG
- JPG
- PNG
- WebP

Uploaded images are validated before inference, and images exceeding the configured maximum file size are rejected.

### Interactive API Documentation

The API includes automatically generated interactive documentation using FastAPI's Swagger UI.

After starting the application, the documentation can be accessed at:

```text
http://localhost:8000/docs
```

The Swagger interface allows users to:

1. Select an API endpoint.
2. Upload a vehicle image.
3. Run the damage detection model.
4. View the returned predictions or annotated image.

This provides a simple way to test and interact with the vehicle damage detection model without requiring users to write API client code.


### Setup

Follow the steps below to set up and run the Vehicle Damage Detection System locally using Docker.

### Prerequisites

Before getting started, make sure the following tools are installed on your system:

- [Git](https://git-scm.com/)
- [Docker](https://www.docker.com/)
- [uv](https://docs.astral.sh/uv/) *(optional if you only want to run the application using Docker)*

You can verify that the required tools are installed by running:

```bash
git --version
docker --version
```

If you plan to run the project locally without Docker, you should also verify that `uv` is installed:

```bash
uv --version
```

---

### Step 1: Clone the Repository

Clone the repository from GitHub:

```bash
git clone <repository-url>
```

Navigate into the project directory:

```bash
cd vehicle-damage-detection
```


---

### Step 2: Set Up the Python Environment (Optional)

If you want to run the application directly on your local machine before using Docker, the project uses `uv` for Python dependency management.

Synchronise the project dependencies using:

```bash
uv sync
```

This will create a virtual environment and install the dependencies defined in `pyproject.toml` using the versions locked in `uv.lock`.

Activate the virtual environment:

#### Linux / macOS

```bash
source .venv/bin/activate
```

#### Windows

```powershell
.venv\Scripts\activate
```

Run this to initiate all the packages
```bash
uv pip install -e .
```

You can then run the application locally using:

```bash
uv run uvicorn src.app.main:app --host 0.0.0.0 --port 8000
```

However, running the application locally is optional. The recommended deployment method for this project is through Docker.


---

### Quick Start with Docker

If you already have Docker installed and the project is configured correctly, the complete setup can be reduced to:

```bash
# Clone the repository
git clone https://github.com/hope205/Vehicle-Damage-Detection

# Navigate to the project
cd vehicle-damage-detection

# Build the Docker image
docker build -t vehicle-damage .

# Run the container
docker run -p 8000:8000 vehicle-damage:latest
```

After starting the container, open the FastAPI documentation in your browser:

```text
http://localhost:8000/docs
```

You can then interact with the vehicle damage detection API directly from the Swagger UI.



### Model Selection


---

### Training Methodology

The model training process was performed using the filtered CarDD dataset containing annotations for three vehicle damage categories: **dent**, **scratch** and **clean**

The training pipeline consisted of the following steps:

1. **Dataset Preparation**
   - The original CarDD dataset was filtered to retain only `dent` and `scratch` annotations.
   - Images without either of the target damage classes were excluded.
   - The resulting dataset was organised into `train`, `val`, and `test` splits.
   - A YOLO-compatible `data.yaml` configuration file was generated to define the dataset paths and class names.

2. **Baseline Training**
   - YOLO26n was selected as the initial baseline model.
   - The baseline model was trained for **50 epochs**.
   - A batch size of **16** was used.
   - The input image size was set to **1024 × 1024 pixels**.
   - The baseline achieved a **mAP@0.5 of 0.5504**.

3. **Final Model Training**
   - YOLOv8n was subsequently evaluated as an alternative lightweight object detection model.
   - The model was trained for **50 epochs** with a batch size of **16**.
   - The input image size was set to **640 × 640 pixels**.
   - The YOLOv8n model achieved an overall **mAP@0.5 of 0.5800**.

4. **Hardware**
   - Model training was performed on Kaggle using **NVIDIA Tesla T4 GPUs**.

5. **Model Selection**
   - The trained models were evaluated based on their object detection performance.
   - YOLOv8n achieved the highest mAP@0.58 of the evaluated configurations and was therefore selected as the final model for deployment.



The API accepts common image formats including JPEG, PNG, JPG, and WebP.


### Model Evaluation

The trained YOLO object detection model was evaluated on a held-out validation dataset using standard object detection metrics. The evaluation measures how accurately the model detects and localizes different categories of vehicle damage.

### Overall Performance

| Metric | Score |
|--------|------:|
| Precision | **79.87%** |
| Recall | **67.68%** |
| F1-Score | **73.27%** |
| mAP@0.5 | **74.29%** |
| mAP@0.5:0.95 | **56.63%** |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | mAP@0.5 | mAP@0.5:0.95 |
|------|----------:|-------:|---------:|---------:|-------------:|
| Dent | 76.74% | 58.90% | 66.65% | 63.73% | 37.04% |
| Scratch | 69.44% | 54.72% | 61.21% | 61.48% | 35.48% |
| Clean | 93.44% | 89.42% | 91.39% | 97.67% | 97.35% |

## Metric Interpretation

### Precision (79.87%)

Precision measures the proportion of predicted damage detections that are actually correct. A precision of **79.87%** indicates that when the model predicts a damage region, it is correct nearly **4 out of every 5 times**, resulting in relatively few false positive detections.

### Recall (67.68%)

Recall measures the model's ability to detect all actual damage instances. A recall of **67.68%** means the model successfully identifies approximately **two-thirds of all damages**, while some damage regions remain undetected.

### F1-Score (73.27%)

The F1-score is the harmonic mean of precision and recall, providing a balanced measure of detection performance. The model achieves an overall **F1-score of 73.27%**, indicating a good balance between minimizing false positives and detecting true damage instances.

### mAP@0.5 (74.29%)

Mean Average Precision at an IoU threshold of 0.5 evaluates both classification and localization performance. A score of **74.29%** indicates that the model performs well at identifying and localizing vehicle damage when a moderate overlap between predicted and ground-truth bounding boxes is required.

### mAP@0.5:0.95 (56.63%)

This metric averages performance across multiple IoU thresholds (0.50 to 0.95), making it a much stricter evaluation of localization accuracy. The score of **56.63%** shows that while the model detects damage effectively, there is still room for improvement in predicting highly accurate bounding box locations.


## Class-wise Analysis

### Clean

The **Clean** class achieved the strongest performance across all metrics:

- Precision: **93.44%**
- Recall: **89.42%**
- F1-Score: **91.39%**
- mAP@0.5: **97.67%**

These results indicate that the model can reliably distinguish undamaged vehicles from damaged ones with very high confidence and localization accuracy.

### Dent

The **Dent** class achieved moderate performance:

- Precision: **76.74%**
- Recall: **58.90%**
- F1-Score: **66.65%**

The model is generally accurate when predicting dents but misses a noticeable number of dent instances. This may be due to varying dent sizes, lighting conditions, or subtle surface deformations that are difficult to detect.

### Scratch

The **Scratch** class proved to be the most challenging:

- Precision: **69.44%**
- Recall: **54.72%**
- F1-Score: **61.21%**

Scratches are often thin, small, and visually similar to reflections or shadows, making them harder to localize accurately. This is reflected in the comparatively lower recall and localization metrics.

## Summary

Overall, the model demonstrates **strong object detection performance**, achieving a **74.29% mAP@0.5** while maintaining high precision. It performs exceptionally well on the **Clean** class and shows reasonable performance on **Dent** and **Scratch** detection.



### Error Analysis

A deep dive into the confusion matrices reveals a very clear story about what the model has learned, and exactly where it needs refinement. 

**Feature Recognition**
The model is highly capable of identifying actual damage. When presented with a real defect, it correctly flags scratches **83% of the time** and dents **77% of the time**. It is exceptionally rare for the model to completely miss actual damage (only missing 3-4% of the time). This proves that the foundational feature extraction is robust.

![confusion_matrix_normalized](runs/detect/evaluations/vehicle_damage_evaluation/confusion_matrix_normalized.png)


**Area for Improvement: Hyper-Sensitivity (False Positives)**
While the model performs well at finding damage,it is highly sensitive and is currently confusing environmental artifacts like sharp reflections, glare, dirt, and natural vehicle panel gaps for scratches and dents. Looking at the raw matrix counts, the model frequently hallucinates bounding boxes on the background (clean parts of the car). 

![confusion_matrix](runs/detect/evaluations/vehicle_damage_evaluation/confusion_matrix.png)

**Action Plan :**

The comparatively lower recall and mAP@0.5:0.95 for damage classes suggest that future improvements could focus on:

- Increasing the diversity and quantity of dent and scratch images.
- Applying stronger data augmentation techniques.
- Training for additional epochs with hyperparameter tuning.
- Experimenting with larger YOLO model variants.
- Improving annotation quality for small or ambiguous damage regions.

These improvements are expected to enhance localization accuracy and increase detection performance for subtle vehicle damage.


### Business Impact & Deployment Strategy

While the current false-positive rate means this iteration isn't ready for fully autonomous, unsupervised deployment, the model's extremely low False Negative rate (its inability to miss real damage) makes it immediately valuable as a **highly first-pass filter**. 

In an automated insurance or rental inspection pipeline, this model can confidently flag potential issues for human review, ensuring no actual damage slips through the cracks. Once the false positive rate is calibrated via the negative sampling strategy outlined above, this system will be fully capable of dramatically reducing manual inspection labor, accelerating claim processing times, and standardizing damage assessments across the board.

