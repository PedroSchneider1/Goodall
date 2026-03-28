# template_features_extractor.py
import os
import numpy as np
import librosa as lib
import func.HelperFunctions as func

from tqdm import tqdm

def calculate_template_features(template_file: str, start_time: float = None, slice_duration: float = None) -> dict:
    """
    Calculate features for a template audio file.

    Args:
        template_file (str): Path to the template audio file.
        start_time (float, optional): Start time for slicing the audio. Defaults to None.
        slice_duration (float, optional): Duration of slices to process. Defaults to None.

    Returns:
        dict: A dictionary containing file_freq, full_cwt_coeffs, full_cwt_freqs, peaks_cwt_features.
    """
    # Load file
    if start_time is not None and slice_duration is not None:
        audio, sr = lib.load(
            template_file,
            offset=start_time,
            duration=slice_duration
        )
    else:
        audio, sr = lib.load(template_file)

    # Compute full CWT transform
    full_cwt_coeffs, full_cwt_freqs = func.cwt_transform(
        audio, sr, wavelet="cmor1.5-1.0"
    )

    # Create wave object
    wave = func.w.Wave(audio, framerate=sr)

    # Find global maximum (loudest sample)
    maxima_sample = np.argmax(wave.ys)

    # Extract CWT features around the peak
    peaks_cwt_features = func.extract_cwt_features(
        audio, sr, [maxima_sample]
    )

    return {
        "full_cwt_coeffs": full_cwt_coeffs,
        "full_cwt_freqs": full_cwt_freqs,
        "peaks_cwt_features": peaks_cwt_features,
    }