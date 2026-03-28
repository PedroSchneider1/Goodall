# decision_trees.py
import numpy as np

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

class DecisionTrees:
    """
    A class to encapsulate both XGBoost and Random Forest models for classification tasks.
    
    Parameters:
        - model_type: str, either 'xgboost' or 'random_forest'
        - **kwargs: additional keyword arguments to pass to the respective model's constructor
    Methods:
        - fit(X_train, y_train): trains the model on the provided training data
        - predict_proba(X_test): returns the predicted probabilities for the positive class
        - predict(X_test, PROBABILITY_THRESHOLD=0.5): returns binary predictions based on the specified probability threshold
        - evaluate(X_test, test_group, cache_test, validation_group): evaluates the model's performance and stores the results in the 'metrics' dictionary
        - summary(): prints a summary of the model's performance metrics
    """
    def __init__(self, model_type: str = 'xgboost', **kwargs):
        self.model_type = model_type
        self.model = None

        self.metrics = dict[str, list[dict[str, float]]]()

        if model_type == 'xgboost':
            self.model = xgb.XGBClassifier(**kwargs)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(**kwargs)
        else:
            raise ValueError("Unsupported model type. Use 'xgboost' or 'random_forest'.")
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        print(f"Training '{self.model_type}' model...")
        self.model.fit(X_train, y_train)
        print("Model training completed.\n")

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X_test)[:, 1]
    
    def predict(self, X_test: np.ndarray, PROBABILITY_THRESHOLD: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X_test)
        return probs > PROBABILITY_THRESHOLD
    
    def evaluate(self, X_test: np.ndarray, test_group: dict, cache_test: dict, validation_group: dict):
        probs = self.predict_proba(X_test)
        detections = self.predict(X_test)

        tp, tn, fn, fp = 0, 0, 0, 0
        fp_list = []

        idx = 0

        for audio in test_group:
            n_peaks = cache_test[audio]['cwt_features'].shape[0]
            for peak_idx in range(n_peaks):
                peak_time = cache_test[audio]['peaks_points'][peak_idx][0]

                is_match = any(
                    abs(peak_time - float(approx)) <= 1
                    for approx in validation_group[audio]
                )

                if detections[idx] and is_match:         # TRUE POSITIVE
                    tp += 1
                elif detections[idx] and not is_match:   # FALSE POSITIVE
                    fp_list.append({
                        "audio": audio,
                        "time": peak_time,
                        "prob": probs[idx]
                    })
                elif not detections[idx] and is_match:   # FALSE NEGATIVE
                    fn += 1
                else:                                    # TRUE NEGATIVE
                    tn += 1

                idx += 1

        positive_detections = np.sum(detections)
        negative_detections = np.sum(~detections)

        fp = len(fp_list)

        total_samples = len(validation_group)
        total_detections = len(detections)

        precision = tp / positive_detections if positive_detections > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        self.metrics.update({
            self.model_type: {
                "total_samples": total_samples,
                "total_detections": total_detections,
                "positive_detections": positive_detections,
                "negative_detections": negative_detections,
                "true_positives": tp,
                "true_negatives": tn,
                "false_negatives": fn,
                "false_positives": fp,
                "precision": precision,
                "sensitivity": sensitivity,
                "specificity": specificity
            }
        })

    def _build_summary(self) -> str:
        m = self.metrics[self.model_type]
        lines = [
            f"Model type:    {self.model_type}",
            "=" * 100,
            f"Total samples:       {m['total_samples']}",
            f"Total detections:    {m['total_detections']}",
            "",
            f"Positive detections: {m['positive_detections']} out of {m['total_detections']}",
            f"Negative detections: {m['negative_detections']} out of {m['total_detections']}",
            "",
            f"True positives:  {m['true_positives']}",
            f"True negatives:  {m['true_negatives']}",
            f"False negatives: {m['false_negatives']}",
            f"False positives: {m['false_positives']}",
            "",
            f"Precision:   {m['precision']   * 100:.2f}%",
            f"Sensitivity: {m['sensitivity'] * 100:.2f}%",
            f"Specificity: {m['specificity'] * 100:.2f}%",
            "=" * 100,
        ]
        return "\n".join(lines)

    def summary(self):
        print(self._build_summary())