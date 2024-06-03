# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 17:55:31 2024

@author: HarisuShehu
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Function to plot and save an image without title
def plot_and_save_image(image, filename):
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.close()

# Function to plot and save a histogram without title
def plot_and_save_histogram(image, filename):
    plt.hist(image.ravel(), 256, [0, 256])
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.savefig(filename)
    plt.close()

# Load the image (assuming the image is already in grayscale)
image_path = "../data/Sample/sample.png"  # Update with your image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded correctly
if image is None:
    raise ValueError("Image not loaded correctly. Please check the file path.")

# Convert image to uint8 if it's not already
if image.dtype != np.uint8:
    image = image.astype(np.uint8)

# Create output directory
output_dir = '../data/images_output'
os.makedirs(output_dir, exist_ok=True)

# Save the original image
plot_and_save_image(image, os.path.join(output_dir, 'original_image.png'))

# Calculate and save the histogram of the original image
plot_and_save_histogram(image, os.path.join(output_dir, 'original_histogram.png'))

# Perform histogram equalization
equalized_image = cv2.equalizeHist(image)

# Save the equalized image
plot_and_save_image(equalized_image, os.path.join(output_dir, 'equalized_image.png'))

# Calculate and save the histogram of the equalized image
plot_and_save_histogram(equalized_image, os.path.join(output_dir, 'equalized_histogram.png'))

print(f"All images and histograms have been saved to {output_dir}")

