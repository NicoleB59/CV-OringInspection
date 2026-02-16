import cv2
import numpy as np
import time


def main():
    # Path to the image you want to load
    img_path = "data/images"

    # Read the image and loads the image aas a 3 channel BGR image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # if the image is not found or failed to load,
    # the image will be null.
    if img is None:
        raise FileNotFoundError(f"Coould not read image")

    # Convert the image to a greyscale
    # greyscalling is easier for thresholding and histogram
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Display the grayscale image in a window titled "gray"
    cv2.imshow("Gray", gray)

    # wait until a key is pressed
    cv2.waitKey(0)

    # close all opencv windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
