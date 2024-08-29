import cv2
import argparse
from tracker import ObjectTracker
from constants import (
    COLOR_PRIMARY_TOLERANCE,
    COLOR_SECONDARY_TOLERANCE,
    MINIMUM_PATH_LENGTH,
    THRESH
)


def main(video_path: str):
    # Create the video capture object
    video_capture = cv2.VideoCapture(video_path)

    # Initialize the ObjectTracker with the video capture and other parameters
    tracker = ObjectTracker(
        video_capture,
        color_primary_tolerance=COLOR_PRIMARY_TOLERANCE,
        color_secondary_tolerance=COLOR_SECONDARY_TOLERANCE,
        min_path_length=MINIMUM_PATH_LENGTH,
        thresh=THRESH
    )
    
    # Process the video and save the object paths to a CSV file
    tracker.process_video()
    tracker.save_paths_to_csv('objects_coordinates.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object detection in a video file.")
    parser.add_argument("--capture", type=str, required=True, help="Path to the video file.")
    args = parser.parse_args()
    
    main(args.capture)