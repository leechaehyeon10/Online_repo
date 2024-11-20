# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 13:40:20 2024

@author: cic
"""
import cv2
import os
import numpy as np

# Parameters for drawing
drawing = False  # True if the mouse is pressed
ix, iy = -1, -1  # Initial x, y coordinates of the region

# List to store segmentation points
annotations = []

# Mouse callback function to draw contours
def draw_contour(event, x, y, flags, param):
    global ix, iy, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        annotations.append([(x, y)])  # Start a new contour

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Add points to the current contour
            annotations[-1].append((x, y))

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # Close the contour by connecting the last point to the first
        annotations[-1].append((x, y))

# Function to calculate bounding box (x, y, H, w)
def get_bounding_box(contour):
    # Convert the contour into a NumPy array
    contour_array = np.array(contour, dtype=np.int32)
    
    # Get bounding box using OpenCV function
    x, y, w, h = cv2.boundingRect(contour_array)  # (x, y) is the top-left, w is width, h is height
    return (x, y, h, w)

# Function to display the image and collect annotations
def segment_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found!")
        return

    # Create a clone of the image for annotation display
    annotated_image = image.copy()
    cv2.namedWindow("Image Segmentation")
    cv2.setMouseCallback("Image Segmentation", draw_contour)

    while True:
        # Show the annotations on the cloned image
        temp_image = annotated_image.copy()
        for contour in annotations:
            points = np.array(contour, dtype=np.int32)
            cv2.polylines(temp_image, [points], isClosed=True, color=(0, 255, 0), thickness=2)
            
            # Get the bounding box (x, y, H, w) for each contour
            x, y, h, w = get_bounding_box(contour)
            cv2.rectangle(temp_image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Draw bounding box

        # Display the image with annotations
        cv2.imshow("Image Segmentation", temp_image)
        
        # Press 's' to save annotations, 'c' to clear, and 'q' to quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            # Save annotations
            with open("annotations.txt", "w") as f:
                for contour in annotations:
                    # Calculate the bounding box for each contour and save
                    x, y, h, w = get_bounding_box(contour)
                    f.write(f"({x}, {y}, {h}, {w})\n")
            print("Annotations saved to annotations.txt")
        elif key == ord("c"):
            # Clear annotations
            annotations.clear()
            annotated_image = image.copy()
            print("Annotations cleared")
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    PathNames = r"D:\02_Lectures\2024_2nd\Lecture_Materials\SW_Dev\Project\val2017\val2017"
    FIleNames = os.listdir(PathNames)
    
    # Find all image files with extensions '.jpg' or '.png'
    FileNames = [_ for _ in FIleNames if _.lower().endswith(('jpg', 'png'))]

    for file_name in FileNames:
        print(f"Processing file: {file_name}")
        # Join the path properly using os.path.join
        image_path = os.path.join(PathNames, file_name)
        segment_image(image_path)