# feature_extraction.py
import os
import time
import numpy as np
import librosa as lib

from func.TemplateFeaturesExtractor import calculate_template_features
from func.HelperFunctions import create_wave, count_local_peaks, extract_cwt_features

import settings

import time

def extract_features(noise_factor: float = 20,
                     num_slices: int = 15,
                     wavelet: str = 'cmor1.5-1.0',
                     test_group: dict = [], train_group: dict = [], templates_group: dict = []):
    """
    Extracts features from audio files in the test, train, templates, and validation groups.

    Args:
        noise_factor (float, optional): Factor to determine the amplitude threshold for peak detection. Defaults to 20.
        num_slices (int, optional): Number of slices to divide the audio into for peak detection. Defaults to 15.
        wavelet (str, optional): Type of wavelet to use for CWT feature extraction. Defaults to 'cmor1.5-1.0'.
        test_group (dict, optional): Dictionary containing test audio file paths and their approximations. Defaults to [].
        train_group (dict, optional): Dictionary containing training audio file paths and their approximations. Defaults to [].
        templates_group (dict, optional): Dictionary containing template audio file paths and their approximations. Defaults to [].

    Returns:
        cache_test (dict): Dictionary containing extracted features for test audio files.
        cache_train (dict): Dictionary containing extracted features for training audio files.
        cache_templates (dict): Dictionary containing extracted features for template audio files.
    """

    FILES_FOLDER_PATH    = settings.filesList[settings.Files.FILES_FOLDER_PATH.value]
    TEMPLATES_CACHE_FILE = settings.filesList[settings.Files.TEMPLATES_CACHE_FILE.value]
    TRAIN_CACHE_FILE     = settings.filesList[settings.Files.TRAIN_CACHE_FILE.value]
    TEST_CACHE_FILE      = settings.filesList[settings.Files.TEST_CACHE_FILE.value]

    timings = {}        # collects per-stage and per-file times
    t_total = time.perf_counter()

    # =================================
    # Templates
    # =================================
    t0 = time.perf_counter()

    if os.path.exists(TEMPLATES_CACHE_FILE):
        print(f"Loading cached templates from {TEMPLATES_CACHE_FILE}")
        cache_templates = np.load(TEMPLATES_CACHE_FILE, allow_pickle=True)
        cache_templates = {k: cache_templates[k].item() for k in cache_templates.files}
        timings['templates'] = {'source': 'cache', 'elapsed_s': round(time.perf_counter() - t0, 3)}
    else:
        print("No cache found, processing templates...")
        cache_templates  = {}
        file_times       = {}

        for file in templates_group:
            tf = time.perf_counter()
            print("Template:", os.path.basename(file))
            candidate_path = os.path.join(FILES_FOLDER_PATH, file) + '.mp3'

            if os.path.exists(candidate_path):
                print("Found at:", candidate_path)
                for approximation in templates_group[file]:
                    cache_templates[file] = calculate_template_features(
                        candidate_path, start_time=float(approximation) - 0.5, slice_duration=2.0
                    )
            else:
                found = None
                target_basename = os.path.basename(file)
                for root, _, files_walk in os.walk(FILES_FOLDER_PATH):
                    if target_basename in files_walk:
                        found = os.path.join(root, target_basename)
                        break
                    rel_candidate = os.path.join(root, file)
                    if os.path.exists(rel_candidate):
                        found = rel_candidate
                        break
                if found:
                    print("Found at:", found)
                    for approximation in templates_group[file]:
                        cache_templates[file] = calculate_template_features(
                            found, start_time=float(approximation) - 0.5, slice_duration=2.0
                        )
                else:
                    print("File not found under", FILES_FOLDER_PATH, ":", file)

            file_times[os.path.basename(file)] = round(time.perf_counter() - tf, 3)

        np.savez_compressed(
            TEMPLATES_CACHE_FILE,
            **{k: np.array(v, dtype=object) for k, v in cache_templates.items()}
        )
        print(f"Cached templates saved to {TEMPLATES_CACHE_FILE}")
        timings['templates'] = {
            'source':    'computed',
            'elapsed_s': round(time.perf_counter() - t0, 3),
            'per_file':  file_times
        }

    # =================================
    # Training data
    # =================================
    t0 = time.perf_counter()

    if os.path.exists(TRAIN_CACHE_FILE):
        print(f"Loading cached training data from {TRAIN_CACHE_FILE}")
        cache_train = np.load(TRAIN_CACHE_FILE, allow_pickle=True)
        cache_train = {k: cache_train[k].item() for k in cache_train.files}
        timings['train'] = {'source': 'cache', 'elapsed_s': round(time.perf_counter() - t0, 3)}
    else:
        cache_train = {}
        file_times  = {}
        stage_times = {}   # breakdown per processing step

        for file in train_group:
            tf = time.perf_counter()
            print("Training data:", os.path.basename(file))
            candidate_path = os.path.join(FILES_FOLDER_PATH, file) + '.mp3'

            if os.path.exists(candidate_path):
                print("Found at:", candidate_path)
                audio, sr = lib.load(candidate_path)
                wave      = create_wave(audio, framerate=sr)

                t_step = time.perf_counter()
                peaks_points = count_local_peaks(wave, num_slices=num_slices, noise_factor=noise_factor)
                t_peaks = round(time.perf_counter() - t_step, 3)

                t_step = time.perf_counter()
                cwt_features = extract_cwt_features(audio, sr, [s[2] for s in peaks_points])
                t_cwt = round(time.perf_counter() - t_step, 3)

                cache_train[file] = {
                    'peaks_points': peaks_points,
                    'cwt_features': cwt_features,
                }
                stage_times[os.path.basename(file)] = {
                    'peak_detection_s': t_peaks,
                    'cwt_extraction_s': t_cwt,
                }

            file_times[os.path.basename(file)] = round(time.perf_counter() - tf, 3)

        np.savez_compressed(
            TRAIN_CACHE_FILE,
            **{k: np.array(v, dtype=object) for k, v in cache_train.items()}
        )
        print(f"All cached results saved to {TRAIN_CACHE_FILE}\n")
        timings['train'] = {
            'source':      'computed',
            'elapsed_s':   round(time.perf_counter() - t0, 3),
            'per_file':    file_times,
            'stage_breakdown': stage_times,
        }

    # =================================
    # Test data
    # =================================
    t0 = time.perf_counter()

    if os.path.exists(TEST_CACHE_FILE):
        print(f"Loading cached test data from {TEST_CACHE_FILE}")
        cache_test = np.load(TEST_CACHE_FILE, allow_pickle=True)
        cache_test = {k: cache_test[k].item() for k in cache_test.files}
        timings['test'] = {'source': 'cache', 'elapsed_s': round(time.perf_counter() - t0, 3)}
    else:
        cache_test  = {}
        file_times  = {}
        stage_times = {}

        for file in test_group:
            tf = time.perf_counter()
            print("Test data:", os.path.basename(file))
            candidate_path = os.path.join(FILES_FOLDER_PATH, file) + '.mp3'

            if os.path.exists(candidate_path):
                print("Found at:", candidate_path)
                audio, sr = lib.load(candidate_path)
                wave      = create_wave(audio, framerate=sr)

                t_step = time.perf_counter()
                peaks_points = count_local_peaks(wave, num_slices=num_slices, noise_factor=noise_factor)
                t_peaks = round(time.perf_counter() - t_step, 3)


                t_step = time.perf_counter()
                cwt_features = extract_cwt_features(audio, sr, [s[2] for s in peaks_points], wavelet=wavelet)
                t_cwt = round(time.perf_counter() - t_step, 3)

                cache_test[file] = {
                    'peaks_points': peaks_points,
                    'cwt_features': cwt_features,
                }
                stage_times[os.path.basename(file)] = {
                    'peak_detection_s':  t_peaks,
                    'cwt_extraction_s':  t_cwt,
                }

            file_times[os.path.basename(file)] = round(time.perf_counter() - tf, 3)

        np.savez_compressed(
            TEST_CACHE_FILE,
            **{k: np.array(v, dtype=object) for k, v in cache_test.items()}
        )
        print(f"All cached results saved to {TEST_CACHE_FILE}\n")
        timings['test'] = {
            'source':          'computed',
            'elapsed_s':       round(time.perf_counter() - t0, 3),
            'per_file':        file_times,
            'stage_breakdown': stage_times,
        }

    # =================================
    # Summary
    # =================================
    timings['total_s'] = round(time.perf_counter() - t_total, 3)

    return cache_test, cache_train, cache_templates, timings