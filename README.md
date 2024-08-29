# Object Detection Application

## Installation

Install all the required packages using requirements.txt from the repository:

```shell
pip install -r requirements.txt
```

## Running app

Open a terminal in your repository root directory and run app.py, passing path to a videocapture in --capture argument

```shell
python app.py --capture luxonis_task_video.mp4
```

|Command line options||

| `--capture` | Path to the video file|

To STOP the program, press `q`


# Overview
This application performs object detection on video files, identifying objects based on their shape, color, and trajectory. The application processes each frame of the video to detect objects, track their paths, and save the detected paths to a CSV file.

Three modules:

1) detector.py
2) tracker.py
3) app.py

# Features of ObjectDetector
 1. Applies thresholding to the frame to create a binary image.
 2. Finds contours of the detected objects in the thresholded image.
 3. For each contour:
    - Calculates the center coordinates of the object.
    - Determines the shape of the object based on the contour.
    - Computes the average color of the object in RGB format.
 4. Collects the properties of each detected object into a dictionary and returns a list of these dictionaries.

# Features of ObjectTracker
 1. Processes an entire video file frame by frame.
 2. Processes a single video frame to detect objects, update their paths, and draw the paths on the frame.
 3. Computes the average color by finding the closest cluster center to a given color, using KMeans clustering. Implemented 
 to avoind treating the same objects as distinct ones on different frames due to slight color recognition inaccuracy
 4. Updates the tracking paths of objects based on new coordinates and color information.
 6. Draws the tracked paths of objects on a given video frame.


# Suggestions for debugging

If application outputs are not considered to be relevant, try experimenting with variables saved in constants.py.
E.g. changing thresholding value, if some objects are not detected. 
A higher number of clusters can capture more color variations but may be more sensitive to noise. A lower number may group too many similar colors together.
