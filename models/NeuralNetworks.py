# neural_networks.py
import numpy as np
import librosa

import keras
from keras import layers

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

class AudioPreprocessor:
    """
    Handles NNs features extraction. Decoupled from any model.

    Args:
        audio (ndarray): The raw audio signal.
        sr (int): Sample rate of the audio signal.
    
    Methods:
        compute_mel_spectrogram: Computes a normalized Mel spectrogram for CNN input.
        compute_filterbanks: Computes Mel filterbank features for RNN input.
    """

    def compute_mel_spectrogram(self, audio: np.ndarray, sr: int, n_mels: int = 128, hop_length: int = 512) -> np.ndarray:
        """Returns (n_mels, T, 1) — ready for CNN input."""
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
        return mel_norm[..., np.newaxis]  # (n_mels, T, 1)

    def compute_filterbanks(self, audio: np.ndarray, sr: int, n_mels: int = 40, hop_length: int = 512) -> np.ndarray:
        """Returns (T, n_mels) — ready for RNN input."""
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length)
        return librosa.power_to_db(mel, ref=np.max).T  # (T, n_mels)

class NeuralNetworks:
    """
    A class to encapsulate both CNNs and RNNs models for binary audio classification tasks.

    Args for CNN (model_type='cnn'):
        input_shape (tuple): Shape of the spectrogram input, e.g. (128, 128, 1)
                             where (mel_bands, time_frames, channels)

    Args for RNN (model_type='rnn'):
        input_shape (tuple): Shape of the sequence input, e.g. (128, 40)
                             where (time_steps, mel_features)
    """
    
    SUPPORTED_FEATURES = ('mel_spectrogram', 'mel_filterbanks', 'custom')
    
    def __init__(self, model_type: str = 'cnn', feature_type: str = 'custom', **kwargs):
        """
        Args:
            model_type:    'cnn', 'cnn1d' or 'rnn'
            feature_type:  'mel_spectrogram' (for cnn) | 'mel_filterbanks' (for rnn) | 'custom' (for both)
        """

        # (820, 25, 1) — 25 steps, 1 channel
        # Where 820 is the number of peaks across all test audio files, 25 is the number of features extracted per peak, and 1 is the single channel dimension for CNN input.
        # This shape is returned by using pooling features, which aggregate the CWT features for each peak into a single vector of 25 features. 
        
        if feature_type not in self.SUPPORTED_FEATURES:
            raise ValueError(f"feature_type must be one of {self.SUPPORTED_FEATURES}")
        
        self.model_type   = model_type
        self.feature_type = feature_type
        self.model        = None
        self.history      = None
        self.metrics      = {}

        self.input_shape = kwargs['input_shape']
        self.model = self._build(self.input_shape)

    def _reshape(self, X: np.ndarray) -> np.ndarray:
        """Reshapes input to match the model's expected input shape."""
        if self.model_type in ('cnn', 'cnn1d'):
            return X.reshape(X.shape[0], *self.input_shape)
        return X

    def _build(self, input_shape: tuple) -> keras.Model:
        if self.model_type == 'cnn':
            # Expects (mel_bands, time_frames, 1) — mel_spectrogram or custom 2D
            return keras.models.Sequential([
                layers.Input(shape=input_shape),
                layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),

                layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),

                layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),

                layers.GlobalAveragePooling2D(),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(1, activation='sigmoid')
            ])

        elif self.model_type == 'cnn1d':
            # Expects (time_steps, features, 1) — custom 1D sequence with channel dimension
            return keras.models.Sequential([
                layers.Input(shape=input_shape),
                layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.MaxPooling1D(2),
                layers.Dropout(0.25),

                layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.MaxPooling1D(2),
                layers.Dropout(0.25),

                layers.GlobalAveragePooling1D(),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(1, activation='sigmoid')
            ])

        elif self.model_type == 'rnn':
            # Expects (time_steps, features) — mel_filterbanks or custom 1D sequence
            return keras.models.Sequential([
                layers.Input(shape=input_shape),
                layers.LSTM(128, return_sequences=True),
                layers.Dropout(0.3),
                layers.LSTM(64, return_sequences=False),
                layers.Dropout(0.3),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.4),
                layers.Dense(1, activation='sigmoid')
            ])

        else:
            raise ValueError("Unsupported model type. Use 'cnn' or 'rnn'.")

    def compile(self, learning_rate: float = 0.001, optimizer=None):
        opt = optimizer or keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    def fit(self, X_train, y_train, epochs=75, batch_size=32, validation_data=None, **kwargs):
        print(f"Training '{self.model_type}' model with '{self.feature_type}' features...")
        
        X_train = self._reshape(X_train)
        if validation_data:
            X_val, y_val = validation_data
            validation_data = (self._reshape(X_val), y_val)
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            **kwargs
        )
        print("Model training completed.\n")
        return self.history

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        return self.model.predict(self._reshape(X_test)).flatten()

    def predict(self, X_test: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return self.predict_proba(X_test) > threshold

    def evaluate(self, X_test: np.ndarray, test_group: dict, cache_test: dict, threshold: float = 0.5):
        probs = self.predict_proba(X_test)
        detections = probs > threshold

        tp, tn, fn, fp = 0, 0, 0, 0
        fp_list = []
        idx = 0

        for audio in cache_test:
            n_peaks = cache_test[audio]['cwt_features'].shape[0]
            for peak_idx in range(n_peaks):
                peak_time = cache_test[audio]['peaks_points'][peak_idx][0]

                is_match = any(
                    abs(peak_time - float(approx)) <= 1
                    for approx in test_group[audio]
                )

                if detections[idx] and is_match:
                    tp += 1
                elif detections[idx] and not is_match:
                    fp_list.append({"audio": audio, "time": peak_time, "prob": probs[idx]})
                elif not detections[idx] and is_match:
                    fn += 1
                else:
                    tn += 1

                idx += 1

        positive_detections = int(np.sum(detections))
        negative_detections = int(np.sum(~detections))
        fp = len(fp_list)

        self.metrics[self.model_type] = {
            "feature_type":        self.feature_type,
            "total_samples":       len(test_group),
            "total_detections":    len(detections),
            "positive_detections": positive_detections,
            "negative_detections": negative_detections,
            "true_positives":      tp,
            "true_negatives":      tn,
            "false_negatives":     fn,
            "false_positives":     fp,
            "precision":           tp / positive_detections if positive_detections > 0 else 0,
            "sensitivity":         tp / (tp + fn)           if (tp + fn) > 0           else 0,
            "specificity":         tn / (tn + fp)           if (tn + fp) > 0           else 0,
        }

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