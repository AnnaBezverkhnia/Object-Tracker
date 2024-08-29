from typing import List, Dict
from dataclasses import dataclass, field
import cv2
import numpy as np
import csv


from detector import ObjectDetector

@dataclass
class ObjectTracker:
    video_capture: cv2.VideoCapture 
    color_primary_tolerance: int
    color_secondary_tolerance: int
    min_path_length: int
    thresh: int
    output_file_path: str = 'detected_objecst_path.png'
    object_paths: List[Dict] = field(default_factory=list)

    def __post_init__(self):
        self.detector = ObjectDetector(self.video_capture, thresh=self.thresh)

    def update_paths(self, detected_objects: List[Dict]):
        '''
        Updates the tracking paths for detected objects based on new coordinates and colors.

        This method performs the following tasks:
        1. Extracts color information from the provided list of new object coordinates.
        2. Clusters these colors using KMeans clustering to identify distinct color groups.
        3. Updates the paths of existing object entries based on their shape and color similarity.
        4. Appends new paths for objects not already tracked.

        :param new_coordinates: List of dictionaries where each dictionary represents an object with the following keys:
            - 'shape': The shape of the object (e.g., 'rectangle', 'circle').
            - 'coordinates': The (x, y) coordinates of the object in the frame.
            - 'color': The color of the object in RGB format.
        :type new_coordinates: List[Dict[str, Union[str, Tuple[int, int], List[int]]]]

        :return: None
        '''
        for new_object in detected_objects:
            for object_path in self.object_paths:
                if object_path['shape'] == new_object['shape'] and self.is_color_similar(object_path['color'], new_object['color']):
                    if new_object['coordinates']:
                        object_path['path'].append(new_object['coordinates'])
                    break
            else:
                self.object_paths.append({
                    'shape': new_object['shape'],
                    'path': [new_object['coordinates']],
                    'color': new_object['color']
                })

    
    def is_color_similar(self, color1, color2, primary_tolerance=10, secondary_tolerance=40):
        '''
        Determines if two colors are similar based on flexible tolerance logic.

        Args:
        - color1, color2: Colors to compare, in BGR format.
        - primary_tolerance: Tolerance for the two closest color components.
        - secondary_tolerance: Tolerance for the third component.

        Returns:
        - True if colors are considered similar, otherwise False.
        '''
        # Calculate the absolute difference for each BGR component
        diff = np.abs(np.array(color1) - np.array(color2))
        
        # Find the two smallest differences and the largest one
        sorted_diff = np.sort(diff)
        
        # Check if the two smallest differences are within the primary tolerance
        if sorted_diff[0] <= primary_tolerance and sorted_diff[1] <= primary_tolerance:
            # Check if the largest difference is within the secondary tolerance
            if sorted_diff[2] <= secondary_tolerance:
                return True
        
        return False
        

    def draw_paths(self, frame: np.ndarray):
        '''Draws the tracked paths on the given frame.

        :param frame: The frame on which paths will be drawn.
        '''
        for object_path in self.object_paths:
            path = object_path['path']
            color = object_path['color']
            for i in range(1, len(path)):
                cv2.line(frame, path[i-1], path[i], color, 2)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        '''Processes a single frame, detects objects, updates paths, and draws object paths on frame.

        :param frame: The frame to be processed.
        :returns: The processed frame with drawn paths.
        '''
        detected_objects = self.detector.detect_objects(frame)
        self.update_paths(detected_objects)
        self.draw_paths(frame)
        return frame

    def process_video(self) -> None:
        '''Processes the entire video, detecting and tracking objects frame by frame.'''
        frames = self.detector.read_frames()
        processed_frame = None
        for frame in frames:
            processed_frame = self.process_frame(frame)
            cv2.imshow("Processed Frame", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.detector.close_capture()
        if processed_frame is not None:
            cv2.imwrite(self.output_file_path, processed_frame)

    def save_paths_to_csv(self, filename: str) -> None:
        '''
        Saves the tracked paths of detected objects to a CSV file.

        The method includes a check to ensure that only objects with paths
        containing at least a minimum number of coordinates are saved. This
        helps in filtering out objects with insufficient tracking data,
        which could be caused by detection inaccuracy

        :param filename: The name of the CSV file where the object paths will be saved.
        :type filename: str
        :return: None
        '''
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['shape', 'color', 'path']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for object_path in self.object_paths:
                coordinates = object_path['path']
                if len(coordinates) >= self.min_path_length:
                    writer.writerow({
                        'shape': object_path['shape'],
                        'color': object_path['color'],
                        'path': object_path['path']
                    })
