from typing import Any

def calculate_f1(precision: float, recall: float) -> float:
    """Calculate F1 score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def extract_overall_metrics(results: Any) -> dict:
    """Extract overall detection metrics."""
    box_precision = float(results.box.mp)
    box_recall = float(results.box.mr)
    box_metrics = {
        "precision": box_precision,
        "recall": box_recall,
        "f1_score": calculate_f1(box_precision, box_recall),
        "map50": float(results.box.map50),
        "map50_95": float(results.box.map),
    }
    return {
        "box": box_metrics,
    }

def extract_per_class_metrics(results: Any, class_names: dict[int, str]) -> dict:
    """Extract per-class detection metrics."""
    per_class_metrics = {}
    box_precision = results.box.p
    box_recall = results.box.r
    box_map50 = results.box.ap50
    box_map50_95 = results.box.all_ap.mean(axis=1)
    for class_id, class_name in class_names.items():
        class_box_precision = float(box_precision[class_id])
        class_box_recall = float(box_recall[class_id])
        class_box_map50 = float(box_map50[class_id])
        class_box_map50_95 = float(box_map50_95[class_id])
        per_class_metrics[class_name] = {
            "box": {
                "precision": class_box_precision,
                "recall": class_box_recall,
                "f1_score": calculate_f1(class_box_precision, class_box_recall),
                "map50": class_box_map50,
                "map50_95": class_box_map50_95,
            },
        }
    return per_class_metrics

def extract_all_metrics(results: Any, class_names: dict[int, str]) -> dict:
    """Extract overall and per-class metrics."""
    return {
        "overall": extract_overall_metrics(results),
        "per_class": extract_per_class_metrics(results, class_names),
    }