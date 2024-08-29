from dataclasses import dataclass
from typing import Tuple, List, Dict, Union
import cv2
import numpy as np
from cv2 import VideoCapture
from cv2.typing import MatLike
import imutils


@dataclass
class ObjectDetector:
    video_capture: VideoCapture
    thresh: int

    def detect_shape(self, contour) -> str:
        '''Detects the shape of an object based on its contour.

        This method uses the contour's perimeter and the number of vertices
        in its approximated polygon to determine if the shape is a rectangle
        or a circle.

        :param contour: The contour of the object.
        :returns: The detected shape as a string ('rectangle' or 'circle').
        '''
        shape = ''
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        if len(approx) == 4:
            shape = 'rectangle'
        else:
            shape = "circle"
        return shape

    def read_frames(self) -> list[MatLike]:
        '''Reads all frames from the video capture.

        This method captures all frames from the video source and stores
        them in a list.

        :returns: List of all frames from the capture.
        '''
        frames = []

        while True:
            ret, frame = self.video_capture.read()
            if not ret:
                break
            frames.append(frame)

        self.close_capture()
        return frames

    
    def frame_thresholding(self, frame: np.ndarray) -> tuple[float, MatLike]:
        '''Applies thresholding to a frame to create a binary image.

        This method:
        - Converts the frame to a grayscale image.
        - Applies Gaussian blur to reduce noise.
        - Applies a binary threshold with a fixed value.

        :param frame: The input frame.
        :returns: The thresholded binary image.
        '''
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, self.thresh, 255, cv2.THRESH_BINARY)[1]
        return thresh

    def get_object_contour(self, thresh: tuple[float, MatLike]) -> tuple:
        '''Finds contours of detected objects in a thresholded image.

        This method extracts the contours of objects from the thresholded
        image.

        :param thresh: The thresholded frame.
        :returns: List of contours represented as numpy arrays.
        '''
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        return contours

    def detect_object_color(self, thresh: tuple[float, MatLike], contour: np.ndarray, frame: np.ndarray) -> MatLike:
        '''
        Detects and returns the average color of a detected object within a given contour in the frame.

        This method performs the following steps:
        1. Creates a mask for the object based on the provided contour.
        2. Converts the frame from BGR color space to HSV color space.
        3. Calculates the mean color within the masked area in the HSV color space.
        4. Converts the mean color from HSV back to BGR color space and returns it.

        :param thresh: Tuple or similar object representing the thresholded image or mask.
        :param contour: Contour array (as detected by `cv2.findContours`) defining the shape of the object.
        :param frame: The original image frame (in BGR color space) from which the object's color is to be detected.
        
        :returns: The mean color of the object in BGR format as a numpy array.
        '''
        object_mask = np.zeros_like(thresh)
        cv2.drawContours(object_mask, [contour], -1, color=255, thickness=cv2.FILLED)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mean_color = cv2.mean(hsv_frame, mask=object_mask)[:3]
        mean_color_bgr = cv2.cvtColor(np.uint8([[mean_color]]), cv2.COLOR_HSV2BGR)[0][0]
        return mean_color_bgr

    def get_object_center_coordinates(self, contour: np.ndarray) -> tuple[int, int] | None:
        '''Calculates the center coordinates of an object's contour.

        This method uses image moments to calculate the centroid of the
        object's contour. If the contour area is zero, it handles the
        ZeroDivisionError and returns None.

        :param contour: The contour of the object.
        :returns: The (x, y) coordinates of the center or None if the area is zero.
        '''
        M = cv2.moments(contour)
        try:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            return cX, cY
        except ZeroDivisionError:
            return None

    def close_capture(self) -> None:
        '''Closes the video capture and destroys all OpenCV windows.

        This method releases the video capture resource and closes any
        OpenCV windows that were opened during the process.
        '''
        self.video_capture.release()
        cv2.destroyAllWindows()


    def detect_objects(self, frame: np.ndarray) -> List[Dict[str, Union[str, Tuple[int, int], List[int]]]]:
        """
        Detects objects in a given frame and returns their properties.

        This method processes the provided frame to detect objects and extract their properties, including shape, center coordinates, and color. It performs the following steps:
        1. Applies thresholding to the frame to create a binary image.
        2. Finds contours of the detected objects in the thresholded image.
        3. For each contour:
            - Calculates the center coordinates of the object.
            - Determines the shape of the object based on the contour.
            - Computes the average color of the object in RGB format.
        4. Collects the properties of each detected object into a dictionary and returns a list of these dictionaries.

        Args:
            frame (np.ndarray): The input frame (image) in which objects are to be detected. It is expected to be a NumPy array representing the image in BGR color space.

        Returns:
            List[Dict[str, Union[str, Tuple[int, int], List[int]]]]:
            A list of dictionaries, each representing a detected object with the following keys:
            - 'shape' (str): The detected shape of the object, which could be 'rectangle' or 'circle'.
            - 'coordinates' (Tuple[int, int]): The (x, y) coordinates of the center of the object.
            - 'color' (List[int]): The average color of the object in RGB format, represented as a list of three integers (R, G, B).

        Example:
            >>> frame = cv2.imread('example_frame.jpg')
            >>> detected_objects = detector.detect_objects(frame)
            >>> print(detected_objects)
            [
                {'shape': 'rectangle', 'coordinates': (200, 150), 'color': [255, 0, 0]},
                {'shape': 'circle', 'coordinates': (400, 300), 'color': [0, 255, 0]}
            ]
        """
        thresh = self.frame_thresholding(frame)
        contours = self.get_object_contour(thresh)
        objects = []

        for contour in contours:
            coordinates = self.get_object_center_coordinates(contour)
            if coordinates is not None:
                cX, cY = coordinates
            else:
                continue
            
            # Determine the shape of the object
            shape = self.detect_shape(contour)
            
            object_color_rgb = self.detect_object_color(thresh, contour, frame).tolist()

            objects.append({
                'shape': shape,
                'coordinates': (cX, cY),
                'color': object_color_rgb
            })
        
        return objects

        