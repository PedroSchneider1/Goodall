# build_dataset.py
import os
import csv
import numpy as np

from models.TemplateMatchingSys import TemplateMatchingSystem

import settings

def build_dataset(train_ratio: float = 0.8, templates_ratio: float = 0.1):
    """
    Builds the dataset by splitting the events into training, testing, templates, and validation groups.

    Args:
        train_ratio (float, optional): _description_. Defaults to 0.8.
        templates_ratio (float, optional): _description_. Defaults to 0.1.

    Returns:
        test_group (dict): Dictionary containing test audio file paths and their approximations.
        train_group (dict): Dictionary containing training audio file paths and their approximations.
        templates_group (dict): Dictionary containing template audio file paths and their approximations.
        validation_group (dict): Dictionary containing validation audio file paths and their approximations.
    """
    EVENTS_CSV = settings.filesList[settings.Files.EVENTS_CSV.value]
    
    # Separate groups for training and testing
    with open(EVENTS_CSV, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Set a fixed random seed for reproducibility
    np.random.seed(42)
    np.random.shuffle(rows)
    
    split_train = int(len(rows) * train_ratio)
    train_rows = rows[:split_train]
    test_rows = rows[split_train:]

    split_templates = int(len(train_rows) * templates_ratio)
    templates_rows = train_rows[:split_templates]
    
    test_group = {}
    train_group = {}
    templates_group = {}
    validation_group = {}

    # Grouping by Approximation - Test, Train, Templates
    for row in test_rows:
        path = row['Media_file']
        approximation = row['Approximation']
        if path not in test_group:
            test_group[path] = []
        test_group[path].append(approximation)

    for row in train_rows:
        path = row['Media_file']
        approximation = row['Approximation']
        if path not in train_group:
            train_group[path] = []
        train_group[path].append(approximation)

    for row in templates_rows:
        path = row['Media_file']
        approximation = row['Approximation']
        if path not in templates_group:
            templates_group[path] = []
        templates_group[path].append(approximation)

    for r in rows:
        path = r['Media_file']
        approximation = r['Approximation']
        if path not in validation_group:
            validation_group[path] = []
        validation_group[path].append(approximation)

    print()

    print(len(test_group), "test events found.")
    print(len(train_group), "train events found.")
    print(len(templates_group), "templates events found.")
    print(len(validation_group), "validation events found.\n")

    return test_group, train_group, templates_group, validation_group

def build_similarities(cache_templates: dict, ALL_AUDIOS: dict) -> tuple:
    """
    Builds similarity matrices between all audios and templates using various similarity metrics.

    Args:
        cache_templates (dict): Cached features for template audio files.
        ALL_AUDIOS (dict): Cached features for all audio files (training and testing).

    Returns:
        tuple: A tuple containing similarity matrices for cosine, euclidean, DWT, wasserstein, and correlation similarities.
    """
    SIMILARITIES_CACHE_FILE = settings.filesList[settings.Files.SIMILARITIES_CACHE_FILE.value]
    similarities_data = np.load(SIMILARITIES_CACHE_FILE, allow_pickle=True) if os.path.exists(SIMILARITIES_CACHE_FILE) else None

    if similarities_data == None:
        # Unwrap targets features
        targets_features = {k: (v["peaks_points"], v["cwt_features"]) for k, v in ALL_AUDIOS.items()}

        templates = []
        for k, v in cache_templates.items():
            templates.append(v["peaks_cwt_features"])

        matcher = TemplateMatchingSystem(normalize_features=True)
        matcher.add_templates(templates)

        cosine_similarity = {}
        correlation_similarity = {}
        euclidean_similarity = {}
        DWT_similarity = {}
        wasserstein_similarity = {}
        for key, items in targets_features.items():
            print("Calculating similarities for:", key)
            if items[1].shape[0] == 0:
                print(f"No CWT features found for {key}, zero-filling similarity matrices.\n")
                cosine_similarity[key] = np.zeros((0, len(templates)))
                correlation_similarity[key] = np.zeros((0, len(templates)))
                euclidean_similarity[key] = np.zeros((0, len(templates)))
                DWT_similarity[key] = np.zeros((0, len(templates)))
                wasserstein_similarity[key] = np.zeros((0, len(templates)))
                continue

            cosine_similarity[key] = matcher.calculate_cosine_similarities(items[1])
            correlation_similarity[key] = matcher.calculate_correlation_similarities(items[1])
            euclidean_similarity[key] = matcher.calculate_euclidean_similarities(items[1])
            DWT_similarity[key] = matcher.calculate_DTW_similarities(items[1])
            wasserstein_similarity[key] = matcher.calculate_wasserstein_similarities(items[1])

        # Save similarities to cache
        np.savez_compressed(
            SIMILARITIES_CACHE_FILE,
            cosine_similarity=cosine_similarity,
            euclidean_similarity=euclidean_similarity,
            DWT_similarity=DWT_similarity,
            wasserstein_similarity=wasserstein_similarity,
            correlation_similarity=correlation_similarity
        )
        print(f"Similarities cached to {SIMILARITIES_CACHE_FILE}")
        
        return (cosine_similarity, euclidean_similarity, DWT_similarity, wasserstein_similarity, correlation_similarity)
    else:
        return (
            similarities_data["cosine_similarity"].item(),
            similarities_data["euclidean_similarity"].item(),
            similarities_data["DWT_similarity"].item(),
            similarities_data["wasserstein_similarity"].item(),
            similarities_data["correlation_similarity"].item()
        )

def prepare_dataset(type: str, cache_templates: dict, cache_train: dict, cache_test: dict, train_group: dict):
    # Previewing the training data
    confusion_matrix = {}
    true_positives = []
    
    # Group all audios info in a single group
    ALL_AUDIOS = {**cache_train, **cache_test}

    # Generate similarity matrices for all audios and templates
    (cosine_similarity, euclidean_similarity, DWT_similarity, wasserstein_similarity, correlation_similarity) = build_similarities(cache_templates,
                                                                                                                                   ALL_AUDIOS)

    # Can use any matrix for counting the number of audios evaluated, since they all have the same keys
    print(f"\n{len(cosine_similarity)} audios evaluated for similarities.")

    # Check which detected peaks in the training group match the approximations in the validation group to build a confusion matrix summary
    for matrix in [cosine_similarity]:
            for file in matrix:
                    for peak_idx in range(matrix[file].shape[0]):
                            try:
                                for approximation in train_group[file]:
                                    if (cache_train[file]['peaks_points'][peak_idx][0] <= float(approximation)+1) and (cache_train[file]['peaks_points'][peak_idx][0] >= float(approximation)-1):
                                        # TP
                                        confusion_matrix.update({'TP': confusion_matrix.get('TP', 0) + 1})
                                        true_positives.append((file, peak_idx))
                                    else:
                                        # FP
                                        confusion_matrix.update({'FP': confusion_matrix.get('FP', 0) + 1})
                            except KeyError:
                                pass
            print()

    print(f"Total number of events: {sum(confusion_matrix.values())}")
    print("Summary:")
    print(confusion_matrix)

    tp_set = set(true_positives)  # Remove duplicates

    print()
    print("="*100)
    print()

    # Building training datasets
    X_all = []
    y_all = []

    if type.upper() == 'POOLING':
        print("Using statistical pooling of similarity metrics for feature extraction.\n")
        
        # Statisctical pooling of similarity metrics
        def pool_templates(metric_values):
            return [
                np.mean(metric_values),
                np.max(metric_values),
                np.std(metric_values),
                np.median(metric_values),
                np.argmax(metric_values),
            ]

        # Building training datasets
        for audio in cache_train:
            n_peaks = cache_train[audio]['cwt_features'].shape[0]

            for peak_idx in range(n_peaks):
                peak_features = []

                for metric in [
                    cosine_similarity,
                    euclidean_similarity,
                    DWT_similarity,
                    wasserstein_similarity,
                    correlation_similarity
                ]:
                    pooled = pool_templates(metric[audio][peak_idx])
                    peak_features.extend(pooled)

                X_all.append(peak_features)
                y_all.append(1 if (audio, peak_idx) in tp_set else 0)

        X_train = np.array(X_all)
        y_train = np.array(y_all)

        # Building testing datasets
        X_test = []

        for audio in cache_test:
            n_peaks = cache_test[audio]['cwt_features'].shape[0]

            for peak_idx in range(n_peaks):
                peak_features = []

                for metric in [
                    cosine_similarity,
                    euclidean_similarity,
                    DWT_similarity,
                    wasserstein_similarity,
                    correlation_similarity
                ]:
                    pooled = pool_templates(metric[audio][peak_idx])
                    peak_features.extend(pooled)

                X_test.append(peak_features)
        X_test = np.array(X_test)
            
    elif type.upper() == 'EACH':
        print("Using each template and each similarity as individual features for feature extraction.\n")
        
        # Using each template and each similarity as individual features
        for metric in [
            cosine_similarity,
            euclidean_similarity,
            DWT_similarity,
            wasserstein_similarity,
            correlation_similarity
        ]:
            for audio in cache_train:
                n_peaks = cache_train[audio]['cwt_features'].shape[0]

                for peak_idx in range(n_peaks):
                    peak_features = metric[audio][peak_idx].tolist()
                    X_all.append(peak_features)
                    y_all.append(1 if (audio, peak_idx) in tp_set else 0)
        X_train = np.array(X_all)
        y_train = np.array(y_all)

        # Building testing datasets
        X_test = []

        for metric in [
            cosine_similarity,
            euclidean_similarity,
            DWT_similarity,
            wasserstein_similarity,
            correlation_similarity
        ]:
            for audio in cache_test:
                n_peaks = cache_test[audio]['cwt_features'].shape[0]

                for peak_idx in range(n_peaks):
                    peak_features = metric[audio][peak_idx].tolist()
                    X_test.append(peak_features)
        X_test = np.array(X_test)
    else:
        print("Invalid feature extraction method specified. Please change the settings to 'POOLING' or 'EACH' inside the configuration snippet.")
        exit(1)

    return X_train, y_train, X_test