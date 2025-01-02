import cv2
import numpy as np
from tqdm import tqdm
import mediapipe as mp
from typing import Tuple
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarkerResult

class DetectionError(Exception):
    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code

    def __str__(self):
        return f"Error Code {self.code}: {self.args[0]}"

class ROIProcessor:
    """
    EXAMPLE USAGE

    Initialization:
        roi = ROIProcessor(video_path: str, config: dict)

    Visualization:
        annotated_frames, fps = roi.detect_with_draw()

    RGB Averaging:
        R_means_aray, L_means_aray, M_means_aray, fps = roi.detect_with_calculate()

    Affine Transformation:
        transformed_frames, fps = roi.detect_with_map()
    """
    ## ROI Landmark Index
    POLYGONS = {
        "R": [139, 100, 203, 135],
        "L": [368, 329, 423, 364],
        "M": [226, 25 , 110, 24 , 23 , 22 , 26 , 112, 243, 244,
                245, 122, 6  , 351, 465, 464, 463, 341, 256, 252,
                253, 254, 339, 255, 446, 265, 372, 345, 352, 280,
                411, 416, 434, 432, 436, 426, 423, 358, 279, 360,
                363, 281, 5  , 51 , 134, 131, 49 , 129, 203, 206,
                216, 212, 214, 192, 187, 50 , 123, 116, 143, 35]
    }

    ## ROI Color
    COLORS = {
        "R": (0, 0, 255),
        "L": (255, 0, 0), 
        "M": (0, 255, 0)
    }

    ## ROI "M": Triangle Indices
    TRIANGLE_INDICES = np.array([
        [214, 212, 216], [214, 216, 207], [192, 214, 207], [187, 192, 207], [187, 207, 205],
        [50 , 187, 205], [50 , 205, 101], [50 , 101, 118], [50 , 118, 117], [123, 50 , 117],
        [116, 123, 117], [116, 117, 111], [143, 116, 111], [143, 111, 35 ], [35 , 111, 31 ],
        [226, 35 , 31 ], [226, 31 , 25 ], [25 , 31 , 228], [31 , 111, 117], [31 , 117, 228],
        [228, 118, 229], [229, 118, 119], [118, 101, 119], [205, 36 , 101], [207, 216, 206],
        [207, 206, 205], [205, 206, 36 ], [206, 203, 36 ], [36 , 203, 129], [36 , 129, 142],
        [36 , 142, 100], [101, 36 , 100], [101, 100, 119], [100, 120, 119], [119, 120, 230],
        [229, 119, 230], [25 , 228, 110], [110, 228, 229], [110, 229, 24 ], [24 , 229, 230],
        [24 , 230, 23 ], [230, 120, 231], [23 , 230, 231], [23 , 231, 22 ], [22 , 232, 26 ],
        [22 , 231, 232], [231, 121, 232], [231, 120, 121], [120, 47 , 121], [120, 100, 47 ],
        [100, 126, 47 ], [100, 142, 126], [142, 209, 126], [142, 129, 209], [129, 49 , 209],
        [49 , 131, 198], [49 , 198, 209], [209, 198, 126], [126, 198, 217], [126, 217, 47 ],
        [47 , 217, 114], [228, 117, 118], [47 , 114, 121], [114, 128, 121], [121, 128, 232],
        [232, 128, 233], [232, 233, 112], [26 , 232, 112], [112, 233, 243], [243, 233, 244],
        [244, 233, 128], [244, 128, 245], [245, 128, 114], [245, 114, 188], [188, 114, 174],
        [114, 217, 174], [217, 198, 174], [198, 236, 174], [198, 131, 134], [198, 134, 236],
        [134, 51 , 236], [236, 51 , 3  ], [174, 236, 196], [236, 3  , 196], [188, 174, 196],
        [188, 196, 122], [245, 188, 122], [122, 196, 6  ], [196, 197, 6  ], [196, 3  , 197],
        [197, 3  , 195], [3  , 51 , 195], [51 , 5  , 195], [6  , 197, 419], [6  , 351, 419],
        [197, 248, 419], [197, 195, 248], [195, 248, 281], [195, 5  , 281], [248, 281, 456],
        [281, 363, 456], [248, 456, 419], [419, 456, 399], [419, 399, 412], [419, 412, 351],
        [351, 412, 465], [465, 412, 343], [465, 343, 357], [465, 357, 464], [464, 357, 453],
        [464, 453, 463], [463, 453, 341], [412, 399, 343], [399, 437, 343], [399, 420, 437],
        [399, 456, 420], [456, 363, 420], [363, 360, 420], [360, 279, 420], [420, 279, 429],
        [437, 420, 355], [420, 429, 355], [437, 355, 277], [343, 437, 277], [343, 277, 350],
        [357, 343, 350], [357, 350, 452], [453, 357, 452], [341, 453, 452], [341, 452, 256],
        [256, 452, 252], [452, 451, 252], [452, 350, 451], [350, 349, 451], [350, 277, 349],
        [277, 329, 349], [277, 355, 329], [355, 371, 329], [355, 429, 371], [429, 358, 371],
        [429, 279, 358], [371, 358, 266], [358, 423, 266], [371, 266, 329], [329, 266, 330],
        [329, 330, 348], [349, 329, 348], [349, 348, 450], [451, 349, 450], [451, 450, 253],
        [252, 451, 253], [266, 423, 426], [266, 426, 425], [266, 425, 330], [425, 426, 427],
        [426, 436, 427], [436, 432, 434], [427, 436, 434], [427, 434, 416], [427, 416, 411],
        [425, 427, 411], [425, 411, 280], [330, 425, 280], [347, 330, 280], [348, 330, 347],
        [346, 347, 280], [346, 280, 352], [346, 352, 345], [450, 348, 449], [449, 348, 347],
        [449, 347, 448], [448, 347, 346], [448, 346, 261], [261, 346, 340], [261, 340, 265],
        [265, 340, 372], [340, 346, 345], [340, 345, 372], [253, 450, 254], [254, 450, 449],
        [254, 449, 339], [339, 449, 448], [339, 448, 255], [255, 448, 261], [255, 261, 446],
        [446, 261, 265]], dtype=np.int32)
    
    ## ROI "M": Coordinates on the Rectangle Mesh that Correspond to Landmark Indices
    INDEX_COORDS = {
        226 : [0  ,0  ], 35  : [0  ,12 ], 143 : [0  ,18 ], 116 : [0  ,24 ],
        123 : [0  ,30 ], 50  : [0  ,36 ], 187 : [0  ,42 ], 192 : [0  ,48 ],
        214 : [0  ,54 ], 212 : [0  ,60 ], 216 : [12 ,60 ], 206 : [18 ,60 ],
        203 : [24 ,60 ], 129 : [30 ,60 ], 49  : [36 ,60 ], 131 : [42 ,60 ],
        134 : [54 ,60 ], 51  : [60 ,60 ], 5   : [66 ,60 ], 281 : [72 ,60 ],
        363 : [78 ,60 ], 360 : [90 ,60 ], 279 : [96 ,60 ], 358 : [102,60 ],
        423 : [108,60 ], 426 : [114,60 ], 436 : [120,60 ], 446 : [132,0  ],
        265 : [132,12 ], 372 : [132,18 ], 345 : [132,24 ], 352 : [132,30 ],
        280 : [132,36 ], 411 : [132,42 ], 416 : [132,48 ], 434 : [132,54 ],
        432 : [132,60 ], 25  : [6  ,0  ], 110 : [12 ,0  ], 24  : [18 ,0  ],
        23  : [24 ,0  ], 22  : [30 ,0  ], 26  : [36 ,0  ], 112 : [42 ,0  ],
        243 : [46 ,0  ], 244 : [50 ,0  ], 245 : [54 ,0  ], 122 : [60 ,0  ],
        6   : [66 ,0  ], 351 : [72 ,0  ], 465 : [78 ,0  ], 464 : [82 ,0  ],
        463 : [86 ,0  ], 341 : [92 ,0  ], 256 : [96 ,0  ], 252 : [102,0  ],
        253 : [108,0  ], 254 : [114,0  ], 339 : [120,0  ], 255 : [126,0  ],
        31  : [6  ,12 ], 111 : [6  ,18 ], 228 : [12 ,12 ], 117 : [12 ,24 ],
        207 : [12 ,44 ], 229 : [18 ,12 ], 118 : [18 ,30 ], 205 : [18 ,40 ],
        230 : [24 ,12 ], 119 : [24 ,24 ], 101 : [24 ,36 ], 36  : [24 ,48 ],
        231 : [30 ,12 ], 120 : [30 ,24 ], 100 : [30 ,36 ], 142 : [30 ,48 ],
        232 : [36 ,10 ], 121 : [36 ,20 ], 47  : [36 ,30 ], 126 : [36 ,40 ],
        209 : [36 ,50 ], 233 : [42 ,10 ], 128 : [42 ,20 ], 114 : [42 ,30 ],
        217 : [42 ,40 ], 198 : [42 ,50 ], 188 : [54 ,15 ], 174 : [54 ,30 ],
        236 : [54 ,45 ], 196 : [60 ,20 ], 3   : [60 ,40 ], 197 : [66 ,20 ],
        195 : [66 ,40 ], 419 : [72 ,20 ], 248 : [72 ,40 ], 412 : [78 ,15 ],
        399 : [78 ,30 ], 456 : [78 ,45 ], 453 : [90 ,10 ], 357 : [90 ,20 ],
        343 : [90 ,30 ], 437 : [90 ,40 ], 420 : [90 ,50 ], 452 : [96 ,10 ],
        350 : [96 ,20 ], 277 : [96 ,30 ], 355 : [96 ,40 ], 429 : [96 ,50 ],
        451 : [102,12 ], 349 : [102,24 ], 329 : [102,36 ], 371 : [102,48 ],
        450 : [108,12 ], 348 : [108,24 ], 330 : [108,36 ], 266 : [108,48 ],
        261 : [126,12 ], 340 : [126,18 ], 448 : [120,12 ], 346 : [120,24 ],
        427 : [120,44 ], 449 : [114,12 ], 347 : [114,30 ], 425 : [114,40 ]}

    def __init__(self, video_path: str, config: dict) -> None:
        """
        Initializes ROIProcessor object, conducts landmark detection on input video upon initialization.

        Args:
            video_path: input video path
            config: configuration dictionary.
        """
        self.video_path = video_path
        self.model_path = config["landmarker_path"]
        self.seg_time_interval = int(config["seg_time_interval"])
        self.detection_result_list = []
        self.frame_list = []
        self.height = 0
        self.width = 0

        self.detection_check = False
        self.detection_count = 0
        self.detection_max = 0

        ## Set options for MediaPipe facelandmark detector
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.VIDEO)
        
        ## Loop through video frames for landmark detection
        with FaceLandmarker.create_from_options(options) as landmarker:
            cap = cv2.VideoCapture(self.video_path)
            ## We take the fps of the input video as "local" fps.
            ## We later interpolate this to fit 30 fps for ppg map.
            self.fps_local = int(cap.get(cv2.CAP_PROP_FPS))
            self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_idx = 0

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                ## Change frame color from BGR to RGB for later visualization.
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.height, self.width, _ = frame.shape
                ## Convert frame image into designated input type for MediaPipe model.
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                timestep_ms= int((frame_idx / self.fps_local) * 1000)
                detection_result = landmarker.detect_for_video(mp_image, timestep_ms)

                if len(detection_result.face_landmarks) == 0:
                    self.detection_count = 0
                else:
                    self.detection_count += 1
                    self.detection_check = True
                    self.detection_max = max(self.detection_max, self.detection_count)

                ## Append detection result and original frame image onto respective lists.
                self.detection_result_list.append(detection_result)
                self.frame_list.append(frame)
                frame_idx += 1

            ## Release resources.
            self.check()
            cap.release()
            cv2.destroyAllWindows()

    def check(self):

        if self.detection_check is False or self.detection_max < self.fps_local * 3:
            raise DetectionError("No Faces Were Detected!", 1004)
        
        succesful_indices = []
        for index, detection_result in enumerate(self.detection_result_list):

            try:
                face_landmark = detection_result.face_landmarks[0]
                landmark_points = [[int(landmark.x * self.width), int(landmark.y * self.height)] for landmark in face_landmark]

                if all(0 <= x <= self.width and 0 <= y <= self.height for x, y in landmark_points):
                    succesful_indices.append(index)

            except IndexError:
                pass

        start = min(succesful_indices)
        end = max(succesful_indices)
        self.frame_list = self.frame_list[start:end]
        self.detection_result_list = self.detection_result_list[start:end]
        self.frame_count = len(self.detection_result_list)

    def map(self, frame: np.ndarray, detection_result: FaceLandmarkerResult) -> np.ndarray:
        """
        Perform affine transformation on a single frame.

        Args:
            frame: Video frame image.
            detection_result: Model inference result for the frame.

        Returns:
            destin_frame: Image with triangular landmarks inside the ROI "M" unwrapped into rectangles.
        """
        ## Return an empty image if no face is detected in the given frame.
        try:
            face_landmark = detection_result.face_landmarks[0]
            origin_frame = np.copy(frame)
            landmark_points = [[int(landmark.x * self.width), int(landmark.y * self.height)] for landmark in face_landmark]
        except Exception:
            return np.zeros((600, 1320, 3), dtype=np.float32)

        ## Create a target image for relocating the landmarks.
        destin_frame = np.zeros((600, 1320, 3), dtype=np.float32)

        ## Apply affine transformation to fit triangular landmarks from the original image into assigned rectangles in the target image.
        for triangle_index in ROIProcessor.TRIANGLE_INDICES:
            ## Coordinates of triangular landmarks in the original image.
            origin_coords = np.array([landmark_points[i] for i in triangle_index], dtype=np.float32)
            ## Coordinates of triangular landmarks in the target image corresponding to the original coordinates.
            destin_coords = np.array([ROIProcessor.INDEX_COORDS[i] for i in triangle_index], dtype=np.float32) * 10

            ## Calculate bounding box coordinates (x, y, w, h) for the triangular landmarks in the original and target images.
            origin_rect = cv2.boundingRect(origin_coords)
            destin_rect = cv2.boundingRect(destin_coords)

            ## Re-align triangular coordinates relative to the top-left corner of the bounding box.
            origin_trig_cropped = [(coord[0] - origin_rect[0], coord[1] - origin_rect[1]) for coord in origin_coords]
            destin_trig_cropped = [(coord[0] - destin_rect[0], coord[1] - destin_rect[1]) for coord in destin_coords]

            ## Crop the bounding box area from the original image.
            ox1, oy1, ox2, oy2 = origin_rect[0], origin_rect[1], origin_rect[0] + origin_rect[2], origin_rect[1] + origin_rect[3]
            cropped_frame = origin_frame[oy1:oy2, ox1:ox2]

            ## Compute the affine transformation matrix to map triangular landmarks from the original to the target image.
            warpMat = cv2.getAffineTransform(np.float32(origin_trig_cropped), np.float32(destin_trig_cropped))
            ## Warp the cropped bounding box from the original image using the affine transformation matrix.
            warped_frame = cv2.warpAffine(cropped_frame, warpMat, (destin_rect[2], destin_rect[3]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
            ## Create a mask of the bounding box size in the target image.
            mask = np.zeros((destin_rect[3], destin_rect[2], 3), dtype=np.float32)
            ## Set the pixel values inside the triangular landmarks in the target mask to 1.
            cv2.fillConvexPoly(mask, np.int32(destin_trig_cropped), (1,1,1), 16, 0)
            ## Apply the mask to the warped bounding box.
            warped_frame = warped_frame * mask

            ## Add the pixel values to the corresponding bounding box area in the target image.
            dx1, dy1, dx2, dy2 = destin_rect[0], destin_rect[1], destin_rect[0] + destin_rect[2], destin_rect[1] + destin_rect[3]
            ## Resize to align dimensions, as slight mismatches may occur during the affine transformation process.
            warped_frame = cv2.resize(warped_frame, (destin_frame[dy1:dy2, dx1:dx2].shape[1], destin_frame[dy1:dy2, dx1:dx2].shape[0]))
            destin_frame[dy1:dy2, dx1:dx2] = destin_frame[dy1:dy2, dx1:dx2] + warped_frame
        return destin_frame

    def draw(self, frame: np.ndarray, detection_result: FaceLandmarkerResult) -> np.ndarray:
        """
        Visualizes a single frame.

        Args:
            frame: Video frame image.
            detection_result: Model inference results for the frame.

        Returns:
            annotated_frame: The frame with ROI polygons for the right, center, and left sides of the face visualized.
        """
        ## If no face is detected in the given frame, return the original frame as is.
        try:
            face_landmark = detection_result.face_landmarks[0]
            annotated_frame = np.copy(frame)
            landmark_points = [[int(landmark.x * self.width), int(landmark.y * self.height)] for landmark in face_landmark]
        except Exception:
            return frame

        ## 프레임 이미지 위에 ROI "R", "L", "M" 경계선 그리기.
        for region, indices in ROIProcessor.POLYGONS.items():
            polygon_points = np.array([landmark_points[i] for i in indices], dtype=np.int32)
            cv2.polylines(annotated_frame, [polygon_points], isClosed=True, color=ROIProcessor.COLORS[region], thickness=2)

        ## 프레임 이미지 위에 ROI "M" 내부 삼각 랜드마크 그리기.
        for indices in ROIProcessor.TRIANGLE_INDICES:
            polygon_points = np.array([landmark_points[i] for i in indices], dtype=np.int32)
            cv2.polylines(annotated_frame, [polygon_points], isClosed=True, color=[220,220,220], thickness=1)
        return annotated_frame

    def calculate(self, frame: np.ndarray, detection_result: FaceLandmarkerResult) -> Tuple[np.ndarray]:
        """
        Calculate the RGB mean for a single frame.

        Args:
            frame: Video frame image.
            detection_result: Model inference results for the frame.

        Returns:
            R_mean: RGB mean value for the right cheek ROI.
            L_mean: RGB mean value for the left cheek ROI.
            M_mean: RGB mean value for the middle face ROI.
        """
        ## Return np.nan if no face is detected in the given frame.
        try:
            face_landmark = detection_result.face_landmarks[0]
            landmark_points = [[int(landmark.x * self.width), int(landmark.y * self.height)] for landmark in face_landmark]
        except Exception:
            return  [np.full(3, np.nan) for _ in range(3)]

        ## Define an inner function to apply a mask based on polygon shapes and calculate the mean color of pixels inside the masked area.
        def get_mean_color(landmark_indices):
            points = np.array([landmark_points[i] for i in landmark_indices], dtype=np.int32)
            mask = np.zeros((self.height, self.width), dtype=np.uint8)
            cv2.fillPoly(mask, [points], 255)
            pixels = frame[mask == 255]
            return np.mean(pixels, axis=0) if pixels.size > 0 else [0, 0, 0]

        ## Calculate the RGB mean for each ROI. Each result is a NumPy array with shape (1, 3).
        R_mean = get_mean_color(ROIProcessor.POLYGONS["R"])
        L_mean = get_mean_color(ROIProcessor.POLYGONS["L"])
        M_mean = get_mean_color(ROIProcessor.POLYGONS["M"])
        return R_mean, L_mean, M_mean
    
    def detect_with_map(self) -> Tuple[np.ndarray, int]:
        """
        Processes each frame by dividing the detected facial region into triangular 
        subregions, applying affine transformations, and reconstructing a rectangular ROI for each frame. 
        The processed frames are then padded (if necessary) and reshaped into segments of uniform size.

        Returns:
            transformed_frames: Transformed and segmented frames as a NumPy array of shape (num_segments, segment_size, height, width, channels)
            fps_local: The frames per second (FPS) used for the video.
        """
        transformed_frames = np.empty((self.frame_count, 600, 1320, 3), dtype=np.uint8)
        with tqdm(total=self.frame_count, desc="Processing Frames", unit="frame") as pbar:
            for idx, (frame, detection_result) in enumerate(zip(self.frame_list, self.detection_result_list)):
                # Apply affine transformation to each frame based on detection results.
                transformed_frames[idx] = self.map(frame, detection_result)
                pbar.update(1)

        # Determine the size of each segment (number of frames per segment).
        segment_size = int(self.fps_local * self.seg_time_interval)
        # Calculate padding size to make the frame count divisible by the segment size.
        padding_size = (segment_size - (self.frame_count % segment_size)) % segment_size
        # Calculate the total number of segments after padding.
        segment_num = int((self.frame_count + padding_size) / segment_size)

        # If padding is needed, pad the frames with zeros along the frame dimension.
        if padding_size > 0:
            transformed_frames = np.pad(
                transformed_frames,
                pad_width=((0, padding_size), (0, 0), (0, 0), (0, 0)),
                mode='constant',
                constant_values=0
            )
        # Reshape the frames into segments with a fixed size.
        transformed_frames = transformed_frames.reshape(segment_num, segment_size, 600, 1320, 3)
        # Drop the last segment.
        transformed_frames = transformed_frames[:-1, :, :, :, :]
        # Return the processed and segmented frames along with the local fps value.
        return transformed_frames, self.fps_local

    def detect_with_draw(self) -> Tuple[np.ndarray, int]:
        """
        Visualizes the boundary lines of the face center, left cheek, and right cheek ROIs 
        detected by the Mediapipe FaceLandmark model.

        Returns:
            annotated_frames: An array of frames with visual annotations.
            fps_local: The frames per second (FPS) used for the video.
        """
        annotated_frames = []
        annotated_frames = np.empty((self.frame_count, self.height, self.width, 3), dtype=np.uint8)
        with tqdm(total=self.frame_count, desc="Processing Frames", unit="frame") as pbar:
            for index, (frame, detection_result) in enumerate(zip(self.frame_list, self.detection_result_list)):
                annotated_frames[index] = self.draw(frame, detection_result)
                pbar.update(1)
        return annotated_frames, self.fps_local

    def detect_with_calculate(self) -> Tuple[np.ndarray]:
        """
        processes video frames by masking the regions outside the face center, left cheek, 
        and right cheek regions of interest (ROIs), leaving only the pixels inside the specified ROIs. 
        It then calculates the average pixel values for each RGB channel. Afterward, linear interpolation 
        is applied to handle any missing values, and the result is returned as a tuple of arrays.

        Returns:
            R_means_array: Array of interpolated average red channel values for each segment.
            L_means_array: Array of interpolated average green channel values for each segment.
            M_means_array: Array of interpolated average blue channel values for each segment.
            fps_local: The frames per second (FPS) of the video.
        """
        # Helper function for linear interpolation of missing values in an array.
        def interpolate(array: np.ndarray) -> np.ndarray:
            for row in range(len(array)):
                current_row = array[row]
                mask = np.isnan(current_row)
                if np.any(mask):
                    current_row[mask] = np.interp(
                        np.flatnonzero(mask),
                        np.flatnonzero(~mask),
                        current_row[~mask]
                    )
            return array

        # Lists to store the average RGB values for each region of interest (ROI)
        R_means_array = np.zeros(shape=(self.frame_count, 3))
        L_means_array = np.zeros(shape=(self.frame_count, 3))
        M_means_array = np.zeros(shape=(self.frame_count, 3))

        # Process each frame and calculate the average RGB values for the ROIs
        with tqdm(total=self.frame_count, desc="Processing Frames", unit="frame") as pbar:
            for index, (frame, detection_result) in enumerate(zip(self.frame_list, self.detection_result_list)):
                R_mean, L_mean, M_mean = self.calculate(frame, detection_result)
                # Append the results to the respective lists
                R_means_array[index] = R_mean
                L_means_array[index] = L_mean
                M_means_array[index] = M_mean
                pbar.update(1)

        # Perform linear interpolation to fill in any missing values for the RGB lists
        R_means_array = interpolate(R_means_array.T)
        L_means_array = interpolate(L_means_array.T)
        M_means_array = interpolate(M_means_array.T)

        # Calculate the segment size and padding required to align the data
        segment_size = int(self.fps_local * self.seg_time_interval)
        padding_size = (segment_size - (self.frame_count % segment_size)) % segment_size
        segment_num = int((self.frame_count + padding_size) / segment_size)
        
        # If padding is required, pad the arrays with zeros
        if padding_size > 0:
            R_means_array = np.pad(R_means_array, ((0, 0), (0, padding_size)), mode='constant', constant_values=0)
            L_means_array = np.pad(L_means_array, ((0, 0), (0, padding_size)), mode='constant', constant_values=0)
            M_means_array = np.pad(M_means_array, ((0, 0), (0, padding_size)), mode='constant', constant_values=0)

        # Reshape and transpose the arrays to align with the required segment structure
        R_means_array = R_means_array.reshape(3, segment_num, segment_size).transpose(1, 2, 0)
        L_means_array = L_means_array.reshape(3, segment_num, segment_size).transpose(1, 2, 0)
        M_means_array = M_means_array.reshape(3, segment_num, segment_size).transpose(1, 2, 0)

        # Remove the last segment, which may be incomplete due to padding
        R_means_array = R_means_array[:-1, :, :]
        L_means_array = L_means_array[:-1, :, :]
        M_means_array = M_means_array[:-1, :, :]

        # Return the processed arrays and the FPS
        return R_means_array, L_means_array, M_means_array, self.fps_local