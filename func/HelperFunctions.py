# helper_functions.py
import pywt
import numpy as np
import pandas as pd
import scipy as sp
import os
import time
import json
import models.Wave as w

from sklearn.metrics import roc_auc_score, f1_score
from models.DecisionTrees import DecisionTrees
from models.NeuralNetworks import NeuralNetworks
from func.Visualization import _save_visualizations

def create_wave(audio: np.ndarray, framerate: int) -> w.Wave:
    """
    Creates a wave object from audio data.

    Args:
        audio (ndarray): The audio signal.
        framerate (int): The sample rate of the audio signal.
    Returns:
        Wave: A wave object containing the audio data and sample rate.
    """

    wave = w.Wave(audio, framerate=framerate)
    return wave

def count_local_peaks(wave: w.Wave, num_slices: int = 15, noise_factor: int = 10) -> tuple:
    """
    Counts the number of local peaks in an audio file.
    
    Args:
        wave: An audio wave object containing the audio data.
        num_slices (int, optional): Number of slices to divide the audio into. Default is 15.
        noise_factor (int, optional): Factor to determine the amplitude threshold for peak detection. Default is 10.
    
    Returns:
        tuple: A tuple containing:
            - int: Number of local maxima points.
            - list: List of tuples, each containing the time, amplitude and index of sample of each local maxima point.
    """
    # Calculate total duration of audio
    total_duration = wave.ts[-1] - wave.ts[0]
    # Calculate slice size in seconds
    slice_duration = total_duration / num_slices

    # Initialize lists for plotting
    slice_start_times = []
    slice_end_times = []
    local_maxima_points = []

    # Calculate the standard deviation of the signal
    std_dev = np.std(wave.ys)
    amp_threshold = ((noise_factor/180) + (std_dev * noise_factor))

    # Process each slice
    for i in range(num_slices):
        start_time = i * slice_duration
        end_time = start_time + slice_duration

        # Extract the segment of the waveform
        wave_slice = wave.segment(start=start_time, duration=slice_duration)

        # Find the local maxima point within this slice
        result = np.argmax(wave_slice.ys)

        # Store slice times and maxima
        slice_start_times.append(start_time)
        slice_end_times.append(end_time)
        if wave_slice.ys[result] > amp_threshold:
            # Find where this slice starts in the original wave samples
            start_sample = int(start_time * wave.framerate)
            absolute_sample_idx = start_sample + result
            
            # Store as (time, amplitude, sample_index) tuple
            local_maxima_points.append((wave_slice.ts[result], wave_slice.ys[result], absolute_sample_idx))

    frequencies = estimate_fundamental_freq(wave, local_maxima_points)
    local_maxima_points = [point for i, point in enumerate(local_maxima_points) if 90 <= frequencies[i] <= 150]

    return local_maxima_points

def estimate_fundamental_freq(wave: w.Wave, peaks_points: list) -> list:
    """
    Estimate the fundamental frequency around given peak points in an audio file.

    Args:
        wave: An audio wave object containing the audio data.
        peaks_points (list): List of tuples containing peak points (time, amplitude).

    Returns:
        list: Estimated fundamental frequencies for each peak point.
    """

    frequencies = []
    for peak in peaks_points:        
        # Extract a segment around the peak (0.2s before and 0.2s after)
        start = max(0, peak[0] - 0.2)
        duration = 0.4
        if start + duration > wave.duration:
            duration = wave.duration - start
        segment = wave.segment(start=start, duration=duration)

        # Compute autocorrelation
        corrs = np.correlate(segment.ys, segment.ys, mode='full')
        corrs = corrs[len(corrs)//2:]  # Keep only the second half
        
        # Normalize the autocorrelation
        corrs /= np.max(corrs)
        
        # Find the lag corresponding to the first peak in the autocorrelation
        lag = np.argmax(corrs[150:250]) + 150
        
        # Calculate the fundamental frequency
        period = lag / segment.framerate
        frequency = 1 / period
        frequencies.append(frequency)
        
    return frequencies

def extract_cwt_features(audio: np.ndarray, sr: int, peaks_indices: list, wavelet: str = 'cmor1.5-1.0') -> np.ndarray:
    """
    Applies a wavelet transform to an audio signal.

    Args:
        audio (ndarray): The audio signal.
        sr (int): The sample rate of the audio signal.
        peaks_indices (list): Indices of the peaks (in seconds) in the audio signal.
        wavelet (str): The type of wavelet to use. Default is 'cmor'.

    Returns:
        ndarray: Extracted features from the CWT coefficients around the peaks.
    """
    window_size = int(0.1 * sr)  # 100 ms window
    features = []

    for peak_idx in peaks_indices:
        # Define extraction window
        start_idx = max(0, peak_idx - window_size // 2)
        end_idx = min(len(audio), peak_idx + window_size // 2)
        
        cwt_window = cwt_transform(audio[start_idx:end_idx], sr, wavelet=wavelet)[0]

        feature_vector = []

        # Energy at different scales (frequency bands)
        scale_energies = np.mean(np.abs(cwt_window) ** 2, axis=1)
        feature_vector.extend(scale_energies)

        # Peak positions in time-frequency space
        peak_positions = np.argmax(np.abs(cwt_window), axis=1)
        feature_vector.extend(peak_positions)

        # Statistical moments to capture shape
        feature_vector.append(np.mean(np.abs(cwt_window)))
        feature_vector.append(np.std(np.abs(cwt_window)))
        feature_vector.append(sp.stats.skew(np.abs(cwt_window).flatten()))
        feature_vector.append(sp.stats.kurtosis(np.abs(cwt_window).flatten()))

        features.append(np.array(feature_vector))
        
    return np.array(features) if features else np.array([]).reshape(0, 0)

def cwt_transform(audio: np.ndarray, sr: int, wavelet: str = 'cmor1.5-1.0', scales: list = None):
    """
    Applies a continuous wavelet transform to an audio signal.

    Args:
        audio (ndarray): The audio signal.
        sr (int): The sample rate of the audio signal.
        wavelet (str): The type of wavelet to use. Default is 'cmor'.
        scales (list, optional): List of scales for the CWT. If None, defaults to 1 to 128.

    Returns:
        tuple: The CWT coefficients and the corresponding frequencies.
    """

    if scales is None:
        scales = np.geomspace(1, 1024, num=100)

    cwt_coeffs, frequencies = pywt.cwt(audio, scales, wavelet, sampling_period=1/sr)
    
    return cwt_coeffs, frequencies

def compare_models(X_train: np.ndarray, y_train: np.ndarray, X_test:  np.ndarray, test_group: dict,
    cache_test: dict, validation_group: dict,
    timestamp: str = 'latest',
    timings: dict = None,
    save_results: bool = True
) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per model containing:
    training_time_s, accuracy, precision, sensitivity, specificity, f1, roc_auc
    — ready for plotting.
    """
    report    = [f"Model Comparison Results — {timestamp}\n{'=' * 100}\n"]
    records   = []

    pos_ratio        = y_train.mean()
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    print(f"Positive ratio: {pos_ratio:.4f}\n")
    report.append(f"Positive ratio: {pos_ratio:.4f}\n")

    # =============================
    # Helper function
    # =============================

    def _run_model(m, fit_fn):
        """Fits, evaluates, and records a single model. Returns its metrics dict."""
        t0 = time.perf_counter()
        fit_fn()
        training_time = time.perf_counter() - t0

        m.evaluate(X_test, test_group, cache_test, validation_group)
        met = m.metrics[m.model_type]

        # For AUC we need probabilities — call predict_proba directly
        probs  = m.predict_proba(X_test)
        y_true_full = []
        y_pred_full = probs
        tp, tn, fn, fp = met['true_positives'], met['true_negatives'], met['false_negatives'], met['false_positives']

        idx = 0
        for audio in test_group:
            n_peaks = cache_test[audio]['cwt_features'].shape[0]
            for peak_idx in range(n_peaks):
                peak_time = cache_test[audio]['peaks_points'][peak_idx][0]
                is_match  = any(
                    abs(peak_time - float(a)) <= 1
                    for a in validation_group[audio]
                )
                y_true_full.append(int(is_match))
                idx += 1

        y_true_full = np.array(y_true_full)

        record = {
            'model':            m.model_type,
            'feature_type':     getattr(m, 'feature_type', 'custom'),
            'training_time_s':  round(training_time, 2),
            'precision':        round(met['precision'],   4),
            'sensitivity':      round(met['sensitivity'], 4),   # recall
            'specificity':      round(met['specificity'], 4),
            'f1':               round(f1_score(y_true_full, y_pred_full > 0.5, zero_division=0), 4),
            'roc_auc':          round(roc_auc_score(y_true_full, y_pred_full), 4),
            'false_negatives': fn,
            'y_true': y_true_full,
            'y_prob': y_pred_full,
        }

        m.summary()
        report.append(
            m._build_summary() +
            f"\nTraining time : {training_time:.2f}s"
            f"\nF1            : {record['f1']:.4f}"
            f"\nROC-AUC       : {record['roc_auc']:.4f}\n"
        )
        records.append(record)

    # =============================
    # Decision trees
    # =============================

    print("=== COMPARING DECISION TREE MODELS ===\n")
    report.append("=== COMPARING DECISION TREE MODELS ===\n")

    dt_models = [
        DecisionTrees('xgboost',
                      objective="binary:logistic", eval_metric="logloss",
                      n_estimators=300, max_depth=5, learning_rate=0.05,
                      scale_pos_weight=scale_pos_weight, random_state=42),
        DecisionTrees('random_forest',
                      n_estimators=100, max_depth=5, random_state=42),
    ]
    for m in dt_models:
        _run_model(m, lambda m=m: m.fit(X_train, y_train))

    # =============================
    # Neural networks
    # =============================

    print("\n=== COMPARING NEURAL NETWORK MODELS ===\n")
    report.append("\n=== COMPARING NEURAL NETWORK MODELS ===\n")

    nn_models = [
        NeuralNetworks('cnn1d', feature_type='custom', input_shape=(X_train.shape[1], 1)),
        NeuralNetworks('rnn',   feature_type='custom', input_shape=(X_train.shape[1], 1)),
    ]
    for m in nn_models:
        m.compile(learning_rate=0.001)
        _run_model(m, lambda m=m: m.fit(
            X_train, y_train,
            epochs=75, batch_size=32, validation_split=0.2
        ))

    # =============================
    # Summarize timings
    # =============================

    print("\n=== Feature extraction timing summary ===")
    for stage in ('templates', 'train', 'test'):
        t = timings[stage]
        print(f"  {stage:<12} [{t['source']:<8}]  {t['elapsed_s']:>7.3f}s")
        if t['source'] == 'computed' and 'stage_breakdown' in t:
            for fname, steps in t['stage_breakdown'].items():
                print(f"    {fname}")
                for step, val in steps.items():
                    print(f"      {step:<22} {val:>6.3f}s")
    print(f"  {'TOTAL':<12}            {timings['total_s']:>7.3f}s")
    print()

    # =============================
    # Save reports and results
    # =============================

    results_df = pd.DataFrame(records).set_index('model')

    report.append("\n=== SUMMARY TABLE ===\n")
    report.append(results_df.to_string())

    if save_results:
        output_dir = f"run_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)

        with open(f"{output_dir}/report.txt", 'w') as f:
            f.write("\n".join(report))

        with open(f"{output_dir}/feature_extraction_timings.json", 'w') as f:
            json.dump(timings, f, indent=2)

        results_df.to_csv(f"{output_dir}/results.csv")
        results_df.to_pickle(f"{output_dir}/results.pkl")
        
        _save_visualizations(results_df, records, output_dir, timestamp, timings)

        print(f"All outputs saved to '{output_dir}/'")
        # run_timestamp/
        # ├── report.txt
        # ├── results.csv
        # └── results.pkl
        # ├── training_time.png
        # ├── roc_curves.png
        # ├── false_negatives.png
        # ├── metrics_comparison.png
        # └── results_table.png

    return results_df