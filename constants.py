# COLOR_TOLERANCE defines the tolerance level for color similarity comparisons.
# A lower value means stricter color matching, while a higher value allows for more variation in color.
COLOR_PRIMARY_TOLERANCE = 10
COLOR_SECONDARY_TOLERANCE = 40

# MINIMUM_PATH_LENGTH sets the minimum number of coordinates required to consider an object's path valid.
# If the path of a detected object contains fewer coordinates, it will be ignored to avoid false detections or noise.
MINIMUM_PATH_LENGTH = 5

# THRESH defines the threshold value for binarizing the image during object detection.
# This value is used in the thresholding operation to separate the object from the background in the image.
THRESH = 20