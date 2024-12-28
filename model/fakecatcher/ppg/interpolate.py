import numpy as np
from scipy.fft import fft, ifft

def frequency_resample(signal: np.ndarray, time_interval: int, original_fps: int, target_fps: int) -> np.ndarray:
    """
    Convert a 1D signal to target_fps using frequency-based interpolation.

    Args:
        signal (np.ndarray): Input signal (1D array).
        original_fps (int): Sampling frequency (FPS) of the input signal.
        target_fps (int): Target sampling frequency (FPS).
    
    Returns:
        np.ndarray: Interpolated signal (1D array).
    """
    if original_fps == target_fps:
        # Return the original signal if FPS is the same
        return signal
    
    # 1. Convert to frequency domain using FFT
    fft_signal = fft(signal)

    # 2. Calculate target length
    target_length = int(time_interval * target_fps)

    # 3. Zero Padding or Truncation
    interpolated_fft = np.zeros(target_length, dtype=complex)
    min_length = min(len(fft_signal), target_length)
    interpolated_fft[:min_length // 2] = fft_signal[:min_length // 2]
    interpolated_fft[-min_length // 2:] = fft_signal[-min_length // 2:]

    # 4. Convert back to time domain using IFFT
    interpolated_signal = ifft(interpolated_fft).real

    return interpolated_signal
