# main.py
import settings
import datetime

from func.BuildDataset import build_dataset, prepare_dataset
from func.FeatureExtraction import extract_features
from func.HelperFunctions import compare_models


def main():
    # -------------------------------
    # Configuration                 |
    # -------------------------------
    TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize folders and files paths
    settings.init()

    # Dataset parameters
    TRAIN_RATIO = 0.85
    TEMPLATES_RATIO = 0.08
    FEATURE_MODE = 'POOLING' # 'EACH' or 'POOLING'

    # Feature extraction parameters
    NUM_SLICES = 60
    NOISE_FACTOR = 15
    WAVELET = 'cmor1.5-1.0' # Morlet wavelet with bandwidth=1.5 and center frequency=1.0 (default)

    # -------------------------------
    # Build dataset                 |
    # -------------------------------

    test_group, train_group, templates_group, validation_group = build_dataset(train_ratio=TRAIN_RATIO,
                                                                               templates_ratio=TEMPLATES_RATIO)

    # -------------------------------
    # Feature calculation           |
    # -------------------------------

    cache_test, cache_train, cache_templates, timings = extract_features(noise_factor=NOISE_FACTOR,
                                                                         num_slices=NUM_SLICES,
                                                                         wavelet=WAVELET,
                                                                         test_group=test_group,
                                                                         train_group=train_group,
                                                                         templates_group=templates_group)

    # -------------------------------
    # Data evaluation               |
    # -------------------------------

    X_train, y_train, X_test = prepare_dataset(type=FEATURE_MODE,
                                               cache_templates=cache_templates,
                                               cache_train=cache_train,
                                               cache_test=cache_test,
                                               train_group=train_group)

    # -------------------------------
    # Model Comparison Results      |
    # -------------------------------
    
    compare_models(X_train,
                   y_train,
                   X_test,
                   test_group,
                   cache_test,
                   validation_group,
                   TIMESTAMP,
                   timings)

if __name__ == "__main__":
    main()
