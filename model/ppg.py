import skvideo.io
import numpy as np

from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt, windows
import math


class PPG_G:
    """
    C. Zhao, C.-L. Lin, W. Chen, and Z. Li, “A novel framework for remote
    photoplethysmography pulse extraction on compressed videos,” in The
    IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
    Workshops, June 2018.
    """

    ## ====================================================================
    ## ========================== Initialization ==========================
    ## ====================================================================

    def __init__(self, roi_path):
        """
        ROI 경로로 비디오 정보를 읽어 인스턴스 초기화.
        """
        self.frames, self.fps = self.read_video(roi_path)

    @classmethod
    def from_frames(cls, frames: np.ndarray, fps: float):
        """
        프레임 이미지 배열, FPS가 이미 존재하는 케이스를 위한 초기화 메소드.
        파이썬은 당장 생성자 오버로딩을 가능하게 만들어라!
        """
        obj = cls.__new__(cls)
        obj.frames = frames
        obj.fps = fps
        return obj

    ## ====================================================================
    ## ============================== Utils ===============================
    ## ====================================================================

    def read_video(self, roi_path: str) -> tuple[np.ndarray, int]:
        """
        비디오 읽어서 프레임 단위 이미지 배열, FPS 로드.

        Args:
            roi_path: ROI 비디오 파일 경로.
        
        Returns:
            frames: 프레임 이미지 배열.
            fps: 초당 프레임 레이트.
        """
        ## 비디오 읽고 프레임 이미지 배열로 변환.
        videogen = skvideo.io.vread(roi_path)
        frames = np.array([frame for frame in videogen])

        ## 비디오 메타 데이터 읽고 fps 문자열 정보 추출.
        meta_data = skvideo.io.ffprobe(roi_path)
        fps = meta_data['video']['@avg_frame_rate']

        ## fps 문자열 정보 정수로 변환.
        fps = int(fps.split('/')[0])
        return frames, fps
    
    def diagonal_average(self, matrix: np.ndarray) -> np.ndarray:
        """
        행렬 diagonal average 적용.

        a11 a12 a13   (a11)             / 1
        a21 a22 a23   (a21 + a12)       / 2
        a31 a32 a33   (a31 + a22 + a13) / 3
                      (a32 + a23)       / 2
                      (a33)             / 1

        Args:
            matrix: diagonal averaging 적용할 행렬.

        Returns:
            averages: "/" 형태의 대각선 상의 값들에 대한 평균값 배열.
        """
        ## 행렬 행과 열 개수 확인.
        rows, cols = matrix.shape
        ## 가능한 대각선 경우의 수 계산.
        num_diags = rows + cols - 1
        ## 평균값 저장할 배열 초기화.
        averages = np.zeros(num_diags)

        ## 각 대각선 별로 평균값 계산.
        for diag in range(num_diags):
            elements = []
            for row_idx in range(rows):
                col_idx = diag - row_idx
                if 0 <= col_idx < cols:
                    elements.append(matrix[row_idx, col_idx])
            if elements:
                averages[diag] = np.mean(elements)
        return averages
    
    def dominant_frequency(self, array: np.ndarray) -> float:
        """
        Fast fourier transform을 통해서 시계열 값을 주파수 성분으로 분해했을 때,
        그 진폭이 가장 강한 성분의 주파수 값을 반환.

        Args: 
            array: 고속 푸리에 변환을 적용할 1D 배열.
        
        Returns:
            dominant_frequency: 진폭이 가장 강한 성분의 주파수 값.
        """
        ## 주어진 시퀀스를 주파수 성분별로 분해.
        ## 주파수 성분의 진폭과 위상을 포함하는 복소수 배열
        fourier = fft(array)

        ## 각 주파수 성분에 해당하는 주파수 값을 나타내는 배열
        frequency = fftfreq(n=len(array), d=1/self.fps)

        ## 진폭 계산.
        amplitude = np.abs(fourier)

        ## 진폭이 가장 큰 주파수 성분의 인덱스를 이용해서 dominant frequency 추출.
        dominant_index = np.argmax(amplitude)
        dominant_frequency = frequency[dominant_index]
        return dominant_frequency

    ## ====================================================================
    ## ================ Single-channel Band Pass Filtering ================
    ## ====================================================================

    def extract_G_trace(self) -> np.ndarray:
        """
        ROI 영상 G채널의 프레임 별 평균값 배열 반환.

        Returns:
            raw_G_trace: ROI 영상 G채널의 프레임 별 평균값을 저장한 넘파이 배열. [F]
        """
        ## axis=3 에서 첫번째와 세번째 값 드랍하여 G채널 값만 보존. [F, H, W]
        G_channel = self.frames[..., 1]
        
        ## axis=0 기준으로 평균 계산. [F]
        raw_G_trace = np.mean(G_channel, axis=(1, 2))
        return raw_G_trace

    def filter_G_trace(self, raw_G_trace: np.ndarray) -> np.ndarray:
        """
        Single-channel Band Pass Filtering.

        Args:
            raw_G_trace: ROI 영상 G채널의 프레임 별 평균값을 저장한 넘파이 배열. [F]

        Returns:
            filtered_G_trace: single-channel band-pass filtering 적용한 배열. [F]
        """
        ## Hz 기준으로 컷오프 설정.
        low = 0.8
        high = 5.0

        ## FPS 절반을 기준으로 컷오프 정규화.
        low = low / (0.5 * self.fps)
        high = high / (0.5 * self.fps)

        # Butterworth band-pass filter 적용.
        b, a = butter(N=4, Wn=[low, high], btype='band')
        
        # 양방향 필터링 적용하여 필터링에 의한 위상 왜곡 제거.
        filtered_G_trace = filtfilt(b, a, raw_G_trace)
        return filtered_G_trace

    ## ====================================================================
    ## ======================= SSA and RC Selection =======================
    ## ====================================================================

    def SSA(self, filtered_G_trace: np.ndarray, window_size: int) -> np.ndarray:
        """
        Singular Spectrum Analysis Decomposition.

        Args:
            filtered_G_trace: single-channel band-pass filtering 적용한 배열. [F]
            window_size: 생성되는 한켈 행렬의 행 개수. W

        Returns:
            rc_array: SSA decomposition을 거쳐 얻는 Reconstructed Components.
                      매칭되는 고유값 기준으로 내림차순 정렬된 다차원 배열. [W, F]
        """
        ## Filtered G trace 배열을 한켈 행렬 Y로 변환.
        N = len(filtered_G_trace)
        K = N - window_size + 1
        Y = np.array([filtered_G_trace[i:i + window_size] for i in range(K)]).T

        ## Y에 SVD를 적용해서 [U][Sigma][V]ᵗ 형태로 계산.
        U, Sigma, Vt = np.linalg.svd(Y, full_matrices=False)

        ## 각 [Sigmaᵢ][Uᵢ][Vᵢ]ᵗ에 대해 diagonal averaging을 적용하여 reconstructed component 뽑기.
        rc_dict = {}
        for idx, sigma in enumerate(Sigma):
            rc_component = sigma * np.outer(U[:,idx], Vt[idx,:])
            rc_component = self.diagonal_average(rc_component)
            rc_dict[sigma] = rc_component

        ## 고유값 기준으로 내림차순 정렬.
        rc_dict = dict(sorted(rc_dict.items(), key=lambda item: item[0], reverse=True))
        rc_array = np.array(list(rc_dict.values()))

        ## 상위 10개 이내의 컴포넌트만 반환.
        rc_array = rc_array[:10] if len(rc_array)>10 else rc_array
        return rc_array

    def RC_selection(self, rc_array: np.ndarray, tolerance:float) -> np.ndarray:
        """
        RC Selection.

        Args:
            rc_array: SSA decomposition을 거쳐 얻는 Reconstructed Components.
                      매칭되는 고유값 기준으로 내림차순 정렬된 다차원 배열. [W, F]
            tolerance: dominant frequency 비교에 사용되는 absolute tolerance.

        Returns:
            rc_trace: RC selection을 거쳐 얻은 Reconstructed Components의 element-wise 합. [F]
        """
        ## twice-relationship 대조를 위한 후보 리스트.
        candidates=[]

        ## 후보 리스트에 각 Reconstructed Component의 dominant frequency 순차적으로 저장.
        for rc_component in rc_array:
            candidates.append(self.dominant_frequency(rc_component))

        ## dominant frequency만 비교하여 twice relationship을 만족하는 페어의 인덱스 추출.
        satisfies_twice = set()
        for source_idx, source_freq in enumerate(candidates):
            for target_idx, target_freq in enumerate(candidates):
                if source_idx < target_idx:
                    s_t = np.isclose(source_freq, 2 * target_freq, atol=tolerance)
                    t_s = np.isclose(target_freq, 2 * source_freq, atol=tolerance)
                    if s_t or t_s:
                        satisfies_twice.add(source_idx)
                        satisfies_twice.add(target_idx)

        ## 인덱스를 기준으로 기존 rc_array에서 최종 컴포넌트 추출.
        rc_array = rc_array[list(satisfies_twice)]

        ## element-wise 합으로 rc_trace 계산.
        rc_trace = np.sum(rc_array, axis=0)
        return rc_trace
    
    ## ====================================================================
    ## ========================== Overlap Adding ==========================
    ## ====================================================================

    def overlap_add(self, array: np.ndarray, window_size: int, step_size: int) -> np.ndarray:
        """
        Overlap Adding.

        Args:
            array: overlap adding을 적용할 1차원 배열. [F]
            window_size: 윈도우 크기.
            step_size: 스텝 사이즈.
        
        Returns:
            overlap_sum: 각 윈도우에 Hanning window를 element-wize 곱처리하고,
                         윈도우 분할에 쓰인 스텝사이즈를 재활용하여 원본 배열의 크기에 맞춰 더한 배열. [F]
        """
        ## 입력 배열과 동일한 크기의 배열 0으로 초기화.
        overlap_sum = np.zeros_like(array)
        ## Hanning window 초기화.
        hann_window = np.hanning(window_size)
        
        for start in range(0, len(array) - window_size + 1, step_size):
            ## 마지막 인덱스 설정.
            end = start + window_size
            ## 인덱싱으로 원본 배열에서 윈도우 추출, hanning window와 곱처리.
            windowed_segment = array[start:end] * hann_window
            ## 결과값의 같은 인덱스 범위에 값 추가. 
            overlap_sum[start:end] += windowed_segment
        return overlap_sum

    ## ====================================================================
    ## ===================== SSA and Spectral Masking =====================
    ## ====================================================================

    def instantaeous_HR(self, preliminary: np.ndarray) -> float:
        """
        SSA decompositon, RC selection과 overlap adding으로 얻은 preliminary를 레퍼런스로 재활용.

        Args:
            preliminary: SSA decompositon, RC selection과 overlap adding으로 얻은 배열.

        Returns:
            f_r: 마스킹 기준으로 사용되는 dominant frequency 값.
        """
        ## 윈도우 사이즈 10초, 스텝 사이즈 1초. (프레임 단위)
        window_size =  self.fps * 10
        step_size = self.fps

        ## 윈도우 단위로 Fast Fourrier Transform 적용.
        ## 각각의 윈도우에 대한 dominant frequency 계산 후 평균값 도출.
        freqs = []
        for start in range(0, len(preliminary) - window_size + 1, step_size):
            end = start + window_size
            windowed_segment = preliminary[start:end]
            dominant_frequency = self.dominant_frequency(windowed_segment)
            freqs.append(dominant_frequency)
        f_r = sum(freqs) / len(freqs)
        return f_r

    def spectral_mask(self, rc_array: np.ndarray, f_r: float,
                      window_size: int, step_size: int) -> np.ndarray:
        """
        SSA decomposition으로 얻은 reconstructed component 후보군 배열을
        instantaeous_HR에서 구한 마스킹 기준, f_r을 사용해서 필터링하고
        마지막으로 overlap adding을 적용해서 PPG 시그널 도출.

        Args:
            rc_array: SSA decomposition으로 얻은 reconstructed component 후보군 배열.
            f_r: 마스킹 기준으로 사용되는 dominant frequency 값.
            window_size: 마스킹 윈도우 크기.
            step_size: 마스킹 스텝 크기.
        
        Returns:
            pulse_signal: PPG 시그널.
        """
        ## Hanning window 초기화.
        hann_window = np.hanning(window_size)

        ## 마스킹을 통과하고 overlap adding이 적용된 최종 결과물 저장하기 위한 배열.
        pulse_signal = np.zeros_like(rc_array[0])

        ## 윈도우 단위로 마스킹 적용.
        for start in range(0, len(rc_array[0]) - window_size + 1, step_size):
            ## 마지막 인덱스 설정.
            end = start + window_size
            ## 마스킹 통과한 RC component에 대한 element-wise addition 값 저장할 배열.
            sum = np.zeros(window_size)

            ## rc_array 안에는 최대 10개의 rc_component가 담겨있다.
            for rc_component in rc_array:
                ## 주어진 시작, 끝 인덱스를 사용해 rc_component 슬라이싱.
                windowed_segment = rc_component[start:end]
                ## dominant frequency 계산.
                f_i = self.dominant_frequency(rc_component)
                ## 마스킹 조건을 만족하면 sum에 더해주기.
                if f_r - (window_size / 2) <= f_i <= f_r + (window_size / 2):
                    sum += windowed_segment

            ## Hanning window 적용하기.
            sum = sum * hann_window
            ## 최종 결과 배열에 더해주기.
            pulse_signal[start:end] += sum
        return pulse_signal

    ## ====================================================================
    ## ============================ Execution =============================
    ## ====================================================================

    def compute_signal(self) -> np.ndarray:
        """
        엑조디아.

        Returns:
            pulse_signal: PPG-G 시그널.
        """
        raw_G_trace = self.extract_G_trace()
        filtered_G_trace = self.filter_G_trace(raw_G_trace)
        rc_array = self.SSA(filtered_G_trace, window_size=120)
        rc_trace = self.RC_selection(rc_array, tolerance=0.2)
        preliminary = self.overlap_add(rc_trace, window_size=120, step_size=30)
        f_r = self.instantaeous_HR(preliminary)
        pulse_signal = self.spectral_mask(rc_array, f_r, window_size=120, step_size=30)
        return pulse_signal
        

class PPG_C:
    """
    G. de Haan and V. Jeanne, “Robust pulse rate from chrominance-based
    rPPG,” IEEE Transactions on Biomedical Engineering, vol. 60, no. 10,
    pp. 2878–2886, Oct 2013.
    """

    ## ====================================================================
    ## ========================== Initialization ==========================
    ## ====================================================================

    def __init__(self, roi_path: str):
        """
        ROI 경로로 비디오 정보를 읽어 인스턴스 초기화.
        """
        self.frames, self.fps = self.read_video(roi_path)

    @classmethod
    def from_frames(cls, frames: np.ndarray, fps: float):
        """
        프레임 이미지 배열이 이미 존재하는 케이스를 위한 초기화 메소드.
        """
        obj = cls.__new__(cls)
        obj.frames = frames
        obj.fps = fps
        return obj

    ## ====================================================================
    ## ============================== Utils ===============================
    ## ====================================================================

    def read_video(self, roi_path: str) -> tuple[np.ndarray, int]:
        """
        비디오 읽어서 프레임 단위 이미지 배열, FPS 로드.

        Args:
            roi_path: ROI 비디오 파일 경로.
        
        Returns:
            frames: 프레임 이미지 배열.
            fps: 초당 프레임 레이트.
        """
        videogen = skvideo.io.vread(roi_path)
        frames = np.array([frame for frame in videogen])

        meta_data = skvideo.io.ffprobe(roi_path)
        fps = meta_data['video']['@avg_frame_rate']
        fps = int(fps.split('/')[0])
        return frames, fps

    ## ====================================================================
    ## ========================== Core Methods ============================
    ## ====================================================================

    def extract_mean_rgb(self) -> np.ndarray:
        """
        각 프레임에서 RGB 평균값을 계산하여 반환.

        Returns:
            RGB: (N, 3) 형태의 평균 RGB 배열.
        """
        RGB = []
        for frame in self.frames:
            avg_rgb = np.sum(np.sum(frame, axis=0), axis=0) / (frame.shape[0] * frame.shape[1])
            RGB.append(avg_rgb)
        return np.asarray(RGB)

    def bandpass_filter(self, lpf: float, hpf: float) -> tuple:
        """
        대역 통과 필터 설계.

        Args:
            lpf: 저주파 차단 주파수.
            hpf: 고주파 차단 주파수.
        
        Returns:
            B, A: 필터 계수.
        """
        nyquist_freq = 0.5 * self.fps
        B, A = butter(3, [lpf / nyquist_freq, hpf / nyquist_freq], btype='bandpass')
        return B, A

    def compute_signal(self, lpf=0.7, hpf=2.5, win_sec=1.6) -> np.ndarray:
        """
        CHROM 알고리즘을 사용해 PPG 신호를 계산.

        Args:
            lpf: 저주파 차단 주파수.
            hpf: 고주파 차단 주파수.
            win_sec: 슬라이딩 윈도우 길이 (초).
        
        Returns:
            BVP: PPG 신호.
        """
        # 1. RGB 평균값 추출
        RGB = self.extract_mean_rgb()
        num_frames = RGB.shape[0]

        # 2. Band-pass filtering
        B, A = self.bandpass_filter(lpf, hpf)
        
        # 3. 슬라이딩 윈도우 설정
        win_length = math.ceil(win_sec * self.fps)
        if win_length % 2:
            win_length += 1
        num_windows = math.floor((num_frames - win_length // 2) / (win_length // 2))

        # 4. PPG 신호 계산
        COMPUTED_PPG_SIGNAL = np.zeros((win_length // 2) * (num_windows + 1))
        win_start, win_mid, win_end = 0, int(win_length // 2), win_length

        for _ in range(num_windows):
            rgb_base = np.mean(RGB[win_start:win_end], axis=0)
            rgb_norm = RGB[win_start:win_end] / rgb_base

            Xs = 3 * rgb_norm[:, 0] - 2 * rgb_norm[:, 1]
            Ys = 1.5 * rgb_norm[:, 0] + rgb_norm[:, 1] - 1.5 * rgb_norm[:, 2]

            # 필터 적용
            Xf = filtfilt(B, A, Xs, axis=0)
            Yf = filtfilt(B, A, Ys)

            alpha = np.std(Xf) / np.std(Yf)
            S_window = Xf - alpha * Yf
            S_window *= windows.hann(win_length)

            COMPUTED_PPG_SIGNAL[win_start:win_mid] += S_window[:win_length // 2]
            COMPUTED_PPG_SIGNAL[win_mid:win_end] = S_window[win_length // 2:]
            win_start = win_mid
            win_mid = win_start + win_length // 2
            win_end = win_start + win_length

        return COMPUTED_PPG_SIGNAL
    
import csv

video_path = "video1.mp4"
test_G_signal = PPG_G(video_path).compute_signal()
print("test_G_signal: ", len(test_G_signal))
test_C_signal = PPG_C(video_path).compute_signal()
print("test_C_signal: ", len(test_C_signal))

with open("output_custom_roi_G_signal.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(test_G_signal)

with open("output_custom_roi_C_signal.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(test_C_signal)