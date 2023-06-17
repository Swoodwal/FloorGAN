import numpy as np
import cv2
import json
import matplotlib.pyplot as plt

# Load the JSON file
PATH=#REPLACE WITH PATH TO JSON FILE
with open(PATH) as f:
    data = json.load(f)

# Iterate over the images and their annotations
for image_data in data["images"]:
    image_id = image_data["id"]
    image_filename = image_data["file_name"]
    image_width = image_data["width"]
    image_height = image_data["height"]

    # Find annotations for the current image
    annotations = [annotation for annotation in data["annotations"] if annotation["image_id"] == image_id]

    # Create an empty mask image
    mask = np.zeros((image_height, image_width), dtype=np.uint8)

    # Iterate over the annotations and draw masks
    for annotation in annotations:
        segmentation = annotation["segmentation"][0]
        polygon = np.array(segmentation).reshape((-1, 2)).astype(np.int32)
        cv2.fillPoly(mask, [polygon], color=annotation["category_id"] + 1)  # Assign category ID as the color value

    # Define a color map for visualization
    color_map = plt.get_cmap("tab20")

    # Apply the color map to the mask image
    colored_mask = color_map(mask)

    # Display the colored mask
    plt.imshow(colored_mask)
    plt.axis('off')
    plt.show()