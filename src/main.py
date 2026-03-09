import cv2 as cv
import numpy as np
import os
import time
import matplotlib.pyplot as plt

# HISTOGRAM
def get_histogram(img):
    # I create an array of 256 values because grayscale images
    # contain intensities from 0 to 255
    # Each position will count how many pixels have that intensity
    hist = np.zeros(256, dtype=int)

    # I loop through every pixel in the image.
    # The goal is to measure how often each intensity occurs
    # This is needed because Otsu thresholding requires the histogram
    for value in img.flat:
        hist[value] += 1

    return hist

def save_histogram(hist, name, folder="histograms"):
    # create the histogram folder if it does not exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    # x-axis values from 0 to 255
    x = np.arange(256)
    # plotted the pixel histogram
    plt.figure()
    plt.stem(x, hist)
    plt.title(f"Histogram of {name}")
    plt.xlabel("Intensity value")
    plt.ylabel("Number of pixels")
    plt.xlim([0, 255])
    out_path = os.path.join(folder, name.replace(".jpg", "_hist.jpg"))
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

# OTSU THRESHOLD
def otsu_threshold(hist, total_pixels):
    # Otsus method workss by testing every possible threshold
    # and selecting the one that best separates foreground and background
    # It does this by maximising between-class variance
    sum_all = 0
    # Compute the total weighted sum of intensity values
    # This is needed to compute the foreground mean later
    for i in range(256):
        # Add grey level * number of pixels at that grey level
        sum_all += i * hist[i]
    # Running sum for bg pixels
    sum_background = 0
    # No of background pixels seen so far
    weight_background = 0
    # This will store the best separation found so far
    max_variance = 0
    best_threshold = 0
    # Try every possible threshold value
    for t in range(256):
        # Add current histogram count to background
        weight_background += hist[t]
        # If background has no pixels yet skip
        if weight_background == 0:
            continue
        # Foreground pixels are the remaining ones
        weight_foreground = total_pixels - weight_background
        if weight_foreground == 0:
            break
        # Update running sum for background intensities
        sum_background += t * hist[t]
        # Compute means of background and foreground
        mean_background = sum_background / weight_background
        mean_foreground = (sum_all - sum_background) / weight_foreground
        # Between-class variance measures how well separated
        # the foreground and background intensities are
        variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        # Keep the threshold that maximises this separation
        if variance > max_variance:
            max_variance = variance
            best_threshold = t

    return best_threshold

# THRESHOLD IMAGE
def threshold_image(img, t):
    # Convert grayscale image into binary image
    # Binary images simplify later processing because
    # we only need to distinguish object vs background
    rows, cols = img.shape
    # Create empty binary image
    binary = np.zeros((rows, cols), dtype=np.uint8)
    # Check each pixel
    for i in range(rows):
        for j in range(cols):
            # Pixels darker than threshold are assumed to belong
            # to the O-ring (foreground).
            if img[i, j] < t:
                binary[i, j] = 255
            else:
                binary[i, j] = 0

    return binary

# DILATION
def dilate(img):
    # Get the height and width of the binary image
    rows, cols = img.shape
    # Get the height and width of the binary image
    out = np.zeros_like(img)
    # Loop through each roW skipping the border
    for i in range(1, rows - 1):
        # Loop through each row skipping the border
        for j in range(1, cols - 1):
            # Assume no white neighbour has been found yet
            found_white = False
            # Check the 3x3 neighbourhood around pixel (i, j)
            for y in range(-1, 2):
                for x in range(-1, 2):
                    # If any neighbouring pixel is white
                    if img[i + y, j + x] == 255:
                        # Mark that a white pixel was found
                        found_white = True
            # If any neighbour was white
            if found_white:
                # If any neighbour was white
                out[i, j] = 255
    return out

# EROSION
def erode(img):
    # Get the image size
    rows, cols = img.shape
    # Create an empty output image
    out = np.zeros_like(img)
    # Loop through each row except borders
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            # Assume all neighbours are white
            all_white = True
            # Check the 3x3 neighbourhood around the pixel
            for y in range(-1, 2):
                for x in range(-1, 2):
                    # If any neighbour is black
                    if img[i + y, j + x] == 0:
                        # If any neighbour is black
                        all_white = False
            # Then the pixel should not stay white
            if all_white:
                # If every neighbour was white
                out[i, j] = 255

    return out

# CLOSING
def closing(img):
    # Apply dilation first to expand white regions
    # Then apply erosion to shrink them slightly
    # This helps fill small gaps or holes
    return erode(dilate(img))

# CONNECTED COMPONENTS
def connected_components(binary):
    rows, cols = binary.shape
    # Create a label image to store object IDs
    labels = np.zeros((rows, cols), dtype=int)
    # First label number
    current_label = 1
    # store the size of each component
    sizes = {}

    # loop through every image
    for i in range(rows):
        for j in range(cols):
            # if the pixel is white and not yet labelled
            if binary[i, j] == 255 and labels[i, j] == 0:
                # start the stack for region growing
                stack = [(i, j)]
                # size counter for this component
                size = 0
                # Continue until stack becomes empty
                while len(stack) > 0:
                    # take the last pixel from the stack
                    y, x = stack.pop()
                    # skip if outside image
                    if y < 0 or y >= rows or x < 0 or x >= cols:
                        continue
                    # skip if pixel is background
                    if binary[y, x] == 0:
                        continue
                    # skip if already labelled
                    if labels[y, x] != 0:
                        continue
                    # Assign current label
                    labels[y, x] = current_label
                    # increase the component
                    size += 1
                    # add all neighbouring pixels to the stack
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            # skip the centre pixel
                            if dy == 0 and dx == 0:
                                continue
                            stack.append((y + dy, x + dx))
                # save the final component
                sizes[current_label] = size
                # Move to the next label
                current_label += 1
    return labels, sizes

# LARGEST COMPONENT
def largest_component(labels, sizes):
    # if no components exist
    if len(sizes) == 0:#
        return None
    # Find the label that has the largest size
    biggest_label = max(sizes, key=sizes.get)
    # Create an empty mask image
    mask = np.zeros_like(labels, dtype=np.uint8)
    # set pixels belonging to the largest component to white
    mask[labels == biggest_label] = 255

    return mask

# AREA
def compute_area(mask):
    return int(np.sum(mask == 255))

# PERIMETER
def compute_perimeter(mask):
    rows, cols = mask.shape
    # start perimeter count
    perimeter = 0
    # Loop through image pixels except borders
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            # if the pixel belopnmgs to the object
            if mask[i, j] == 255:
                # if any neighbour is background
                if (mask[i - 1, j] == 0 or
                    mask[i + 1, j] == 0 or
                    mask[i, j - 1] == 0 or
                    mask[i, j + 1] == 0):
                    # The pixel is part of the edge
                    perimeter += 1
    return perimeter

# HOLE CHECK
def has_hole(mask):
    rows, cols = mask.shape
    # Create an empty image that will hold the inverse of the mask
    inverse = np.zeros_like(mask)
    # Loop through every pxiel
    for i in range(rows):
        for j in range(cols):
            # if the current pixel is black in the mask
            # make it white in the inverse image
            if mask[i, j] == 0:
                inverse[i, j] = 255
    # create another image to keep track visited pixels
    visited = np.zeros_like(mask, dtype=np.uint8)
    # create a stack for flood filling
    stack = []
    # Look along the top and bottom border of the inverse image
    for j in range(cols):
        # If the pixel on the top border is white add it to the stack
        if inverse[0, j] == 255:
            stack.append((0, j))
        if inverse[rows - 1, j] == 255:
            stack.append((rows - 1, j))

    for i in range(rows):
        if inverse[i, 0] == 255:
            stack.append((i, 0))
        if inverse[i, cols - 1] == 255:
            stack.append((i, cols - 1))
    # Flood fill all white regiuoisn connected to the image border
    while len(stack) > 0:
        # Take 1 pixel from the stack
        y, x = stack.pop()
        # skip if outside the image
        if y < 0 or y >= rows or x < 0 or x >= cols:
            continue
        # skip if visted
        if visited[y, x] == 1:
            continue
        # skip if pixel is black in the inverse image
        if inverse[y, x] == 0:
            continue
        # mark the current pixel
        visited[y, x] = 1
        # Add its 4 conneted neighbours to the stack
        stack.append((y + 1, x))
        stack.append((y - 1, x))
        stack.append((y, x + 1))
        stack.append((y, x - 1))
    # check if any white pixel in the inverse image
    for i in range(rows):
        for j in range(cols):
            # IMPORTANT
            # if a pixel exists it means there is an enclosed hole
            if inverse[i, j] == 255 and visited[i, j] == 0:
                return True
    return False

# CENTROID
def get_centroid(mask):
    # Get the coordinates of all white pixels in the mask
    ys, xs = np.where(mask == 255)
    # if there are no white pixels no centroid can be found
    if len(xs) == 0:
        return None
    # compute the average x cortd of the white pixel and the same for y
    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    return cx, cy

# RADIAL THICKNESS FEATURES
def radial_features(mask, step=10):
    # find the centre of the ring region
    centre = get_centroid(mask)
    # if the centre does not exist, return large dummy vals
    if centre is None:
        return 999, 999
    # split centre cords into x and y
    cx, cy = centre
    rows, cols = mask.shape
    # Estimate the max distance needed to scan form the centre
    max_r = int(np.sqrt(rows * rows + cols * cols))
    # List to store ring thickness values in different directions
    thicknesses = []
    # Count how many directions fail to detect a proper ring thickness
    missing_angles = 0
    # check around the whole ring in fixed angle steps
    for deg in range(0, 360, step):
        # Convert current angle to radians
        theta = np.deg2rad(deg)
        # store the distances where the ray hits white pixels
        hits = []
        # Move outward from the centre along the current direction
        for r in range(max_r):
            # Compute x and y position along the ray
            x = int(round(cx + r * np.cos(theta)))
            y = int(round(cy + r * np.sin(theta)))
            # Stop of the ray leaves the image
            if x < 0 or x >= cols or y < 0 or y >= rows:
                break
            # if the ray hits a white pixel record the radius
            if mask[y, x] == 255:
                hits.append(r)
        # if we do not detect at least two white point
        # then the ring is missing in this direction
        if len(hits) < 2:
            missing_angles += 1
        # otherwise measure ring thickness in this direction
        else:
            thicknesses.append(hits[-1] - hits[0])
    # if no thickness values were found return dummy vals
    if len(thicknesses) == 0:
        return 999, 999
    # convert the list oif thickness into a Numpy array
    thicknesses = np.array(thicknesses, dtype=np.float32)
    thick_std = float(np.std(thicknesses))
    # return both defect measures
    return missing_angles, thick_std

# CIRCULARITY
def compute_circularity(mask):
    # Compute the area of the object
    area = compute_area(mask)
    # compute the perimeter of the object
    perimeter = compute_perimeter(mask)
    # if the perimeter is zero avoid division by 0
    if perimeter == 0:
        return 0.0
    # using the cirularity formula
    return (4 * np.pi * area) / (perimeter ** 2)

# CLASSIFY
def classify(mask):
    # compute the following features
    area = compute_area(mask)
    perimeter = compute_perimeter(mask)
    circularity = compute_circularity(mask)
    missing_angles, thick_std = radial_features(mask, step=10)
    # if the region is too small it is not a valid o ring
    if area < 500:
        result = "FAIL"
    # if the ring does not contain a hole it is broken
    elif has_hole(mask) is False:
        result = "FAIL"
    # if te ring thickness changes too much it is defective
    elif thick_std > 0.75:
        result = "FAIL"
    # if the shape is not circular enough fail it
    elif circularity < 0.11:
        result = "FAIL"
    # otherwise it is classified as a complete ring
    else:
        result = "PASS"

    return result, area, perimeter, circularity, missing_angles, thick_std

# PROCESS ONE IMAGE
def process_image(path, name, output_folder="results"):
    # start timer
    start = time.time()
    # read image in greyscale
    img = cv.imread(path, 0)
    # if the image could not be loaded stop
    if img is None:
        return
    # build the histgram
    hist = get_histogram(img)
    save_histogram(hist, name)
    # Find an automatic threshold using thres method above
    threshold_value = otsu_threshold(hist, img.size)
    # Convert greyscale image into binary image
    binary = threshold_image(img, threshold_value)
    # clean the binary image using closing
    cleaned = closing(binary)
    # label connected foreground regions
    labels, sizes = connected_components(cleaned)
    # keep only the largest region assumed to be the O ring
    mask = largest_component(labels, sizes)
    # if no region was found stop
    if mask is None:
        return
    # Analyse the sxtracted region and classify it
    result, area, perimeter, circularity, missing_angles, thick_std = classify(mask)
    # stop timer
    end = time.time()
    # convert the greyscale image to colour so text can be written
    output = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    colour = (0, 255, 0) if result == "PASS" else (0, 0, 255)

    y = 25
    step = 25
    # write classification results onto output image
    cv.putText(output, f"Result: {result}", (10, y),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    y += step
    # TIME
    cv.putText(output, f"Time: {round(end - start, 4)} s", (10, y),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    y += step
    # THRESHOLD
    cv.putText(output, f"Threshold: {threshold_value}", (10, y),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    y += step
    # AREA
    cv.putText(output, f"Area: {area}", (10, y),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    y += step
    # PERIMETER
    cv.putText(output, f"Perimeter: {perimeter}", (10, y),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    y += step
    # CIRCULARITY
    cv.putText(output, f"Circularity: {round(circularity, 4)}", (10, y),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    y += step
    # MISSING ANGLES
    cv.putText(output, f"Missing angles: {missing_angles}", (10, y),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    y += step
    # THICKNESS
    cv.putText(output, f"Thickness std: {round(thick_std, 4)}", (10, y),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    save_path = os.path.join(output_folder, name)
    cv.imwrite(save_path, output)

# MAIN
def main():
    folder = "C:/Users/Bulal/PyCharmMiscProject/CV-OringInspection/data/images"

    files = os.listdir(folder)
    files.sort()

    for filename in files:
        if filename.lower().endswith(".jpg"):
            path = os.path.join(folder, filename)
            process_image(path, filename)

if __name__ == "__main__":
    main()