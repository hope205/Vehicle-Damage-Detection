# Vehicle Damage Detection System


## Solution Overview

The proposed solution is an end-to-end computer vision system for detecting two common types of vehicle damage: **scratches and dents**.

The solution follows the following workflow:

1. **Dataset Collection**
   - The CarDD dataset with YOLO annotations was used as the source dataset.

2. **Dataset Exploration and Preprocessing**
   - The original dataset contains multiple vehicle damage categories.
   - The dataset was filtered to retain only the **dent** and **scratch** classes relevant to this task.
   - Images and their corresponding YOLO-format annotations were extracted and organised into training, validation, and test splits.

3. **Model Training**
   - A YOLO-based object detection model was trained on the filtered dataset.
   - The model learns to identify the location and category of visible vehicle damage.

4. **Model Evaluation**
   - The trained model was evaluated using object detection metrics such as:
     - Precision
     - Recall
     - F1-score
     - mAP@50
     - mAP@50-95
   - Error analysis was performed to identify common failure cases and areas for improvement.

5. **Inference API**
   - The trained model was integrated into a **FastAPI** application.
   - The API accepts vehicle images and runs inference using the trained YOLO model.
   - Predictions are returned as structured responses containing detected damage classes, confidence scores, and bounding-box coordinates.

6. **Containerisation**
   - The application is packaged using Docker to provide a consistent and reproducible deployment environment.

### High-Level Workflow

```text
Vehicle Image
      │
      ▼
FastAPI Inference API
      │
      ▼
Image Preprocessing
      │
      ▼
YOLO Object Detection Model
      │
      ├───────────────┐
      ▼               ▼
   Dent           Scratch
      │               │
      └───────┬───────┘
              ▼
     Detection Results
              │
              ▼
     JSON API Response




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

After filtering the dataset to include only dents and scratches, the class distribution was:

| Class | Instances | Percentage |
|---|---:|---:|
| Dent | 2,543 | 41.43% |
| Scratch | 3,595 | 58.57% |
| **Total** | **6,138** | **100%** |

The filtered dataset contains a higher number of **scratch** annotations than **dent** annotations. Scratches account for **58.57%** of all annotated damage instances, while dents account for **41.43%**.

Overall, the class distribution is reasonably balanced, although scratches are more represented than dents. This filtered dataset was subsequently used to train the YOLO-based vehicle damage detection model.




## 7. Preprocessing

## 8. Model Selection

### Baseline


### Candidate Models
### Final Model

## 9. Training Methodology



## 13. API

## 14. Project Structure

## 15. Setup

## 16. Running Inference

base model of yolov8n image size of 640


## 17. Docker

Navigate to the root directy of the application

docker build -t vehicle-damage .

docker run -p 8000:8000 vehicle-damage:latest

## 📊 Evaluation & Metrics

The primary goal of this initial training phase was to determine if the model could successfully identy scratchs and dents features of vehicle damage. The results strongly indicate that it has. 

Our baseline model achieved an overall **mAP@0.5 of 0.580**, with closely balanced performance across our target classes:
* **Dent AP:** 0.585
* **Scratch AP:** 0.575

To optimize the model's performance in a real-world setting, I analyzed the F1-Confidence curve to find the perfect balance between Precision and Recall. The curve peaks at an F1 score of 0.58 when using a confidence threshold of **0.405**. By configuring the model to ignore predictions below this 40.5% confidence mark, we can filter out a significant portion of low-confidence noise.

`[Insert BoxPR_curve.png here]`
`[Insert BoxF1_curve.png here]`

## 🔍 Error Analysis: What the Data is Telling Us

A deep dive into the confusion matrices reveals a very clear story about what the model has learned, and exactly where it needs refinement. 

**The Good: Excellent Feature Recognition**
The model is highly capable of identifying actual damage. When presented with a real defect, it correctly flags scratches **83% of the time** and dents **77% of the time**. It is exceptionally rare for the model to completely miss actual damage (only missing 3-4% of the time). This proves that the foundational feature extraction is robust.

`[Insert confusion_matrix_normalized.png here]`

**Area for Improvement: Hyper-Sensitivity (False Positives)**
While the model is fantastic at finding damage, it is currently "over-eager." Looking at the raw matrix counts, the model frequently hallucinates bounding boxes on the background (clean parts of the car). It is highly sensitive and is currently confusing environmental artifacts—like sharp reflections, glare, dirt, and natural vehicle panel gaps—for scratches and dents. 

In short: The model knows exactly what a scratch looks like, but it hasn't yet learned what a *healthy* car looks like. 

`[Insert confusion_matrix.png here]`

**Action Plan for Version 2.0:**
To easily correct this hyper-sensitivity, the next training iteration will include:
1. Injecting a large volume of "background-only" images (perfectly intact vehicles under various lighting conditions) with empty annotations to force the model to learn what *not* to detect.
2. Isolating the specific reflections and panel gaps that caused the false positives, labeling them as background, and feeding them back into the model.Instead of two classes, we will be having three
3. Applying more random glare and shadow effects during training to make the model blind to lighting artifacts. 

## 💼 Business Impact & Deployment Strategy

While the current false-positive rate means this iteration isn't ready for fully autonomous, unsupervised deployment, the model's extremely low False Negative rate (its inability to miss real damage) makes it immediately valuable as a **highly first-pass filter**. 

In an automated insurance or rental inspection pipeline, this model can confidently flag potential issues for human review, ensuring no actual damage slips through the cracks. Once the false positive rate is calibrated via the negative sampling strategy outlined above, this system will be fully capable of dramatically reducing manual inspection labor, accelerating claim processing times, and standardizing damage assessments across the board.


## 18. Limitations

## 19. Future Improvements