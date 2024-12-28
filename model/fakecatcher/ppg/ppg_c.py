import numpy as np
from scipy.signal import butter, filtfilt, windows
import math


class PPG_C:
    """
    G. de Haan and V. Jeanne, “Robust pulse rate from chrominance-based
    rPPG,” IEEE Transactions on Biomedical Engineering, vol. 60, no. 10,
    pp. 2878–2886, Oct 2013.
    """
    def __init__(self, RGB_mean_array: np.ndarray, fps: int):
        """
        Initialize the instance by reading video information from the ROI path.
        """
        self.RGB_mean_array = RGB_mean_array
        self.fps = fps

    ## ====================================================================
    ## ========================== Core Methods ============================
    ## ====================================================================

    def bandpass_filter(self, lpf: float, hpf: float) -> tuple:
        """
        Design a band-pass filter.

        Args:
            lpf: Low-pass cutoff frequency.
            hpf: High-pass cutoff frequency.
        
        Returns:
            B, A: Filter coefficients.
        """
        nyquist_freq = 0.5 * self.fps
        B, A = butter(3, [lpf / nyquist_freq, hpf / nyquist_freq], btype='bandpass')
        return B, A

    def compute_signal(self, lpf=0.7, hpf=2.5, win_sec=1.6, step_sec=0.7) -> np.ndarray:
        """
        Compute the PPG signal using the CHROM algorithm.

        Args:
            lpf: Low-pass cutoff frequency.
            hpf: High-pass cutoff frequency.
            win_sec: Sliding window length (seconds).
        
        Returns:
            BVP: PPG signal.
        """
        # 1. Extract RGB mean values
        num_frames = self.RGB_mean_array.shape[0]

        # 2. Band-pass filtering
        B, A = self.bandpass_filter(lpf, hpf)
        
        # 3. Set sliding window
        win_length = int(self.fps * win_sec)
        step_size = int(self.fps * step_sec)

        # 4. Compute PPG signal
        COMPUTED_PPG_SIGNAL = np.zeros(num_frames)  # Initialize result array with num_frames size
        win_start = 0

        while win_start + win_length <= num_frames:
            win_end = win_start + win_length

            # Mean and normalization for the current window
            rgb_base = np.mean(self.RGB_mean_array[win_start:win_end], axis=0)
            rgb_base += 1e-6
            rgb_norm = self.RGB_mean_array[win_start:win_end] / rgb_base

            Xs = 3 * rgb_norm[:, 0] - 2 * rgb_norm[:, 1]
            Ys = 1.5 * rgb_norm[:, 0] + rgb_norm[:, 1] - 1.5 * rgb_norm[:, 2]

            # Apply filter
            Xf = filtfilt(B, A, Xs, axis=0)
            Yf = filtfilt(B, A, Ys)

            std_Yf = np.std(Yf)
            if std_Yf > 1e-6:
                alpha = np.std(Xf) / std_Yf
            else:
                alpha = 0.0

            S_window = Xf - alpha * Yf
            S_window *= windows.hann(win_length)

            # Reflect results
            COMPUTED_PPG_SIGNAL[win_start:win_end] += S_window

            # Move to the next window
            win_start += step_size

        return COMPUTED_PPG_SIGNAL