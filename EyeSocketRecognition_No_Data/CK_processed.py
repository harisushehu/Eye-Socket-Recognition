# -*- coding: utf-8 -*-
"""
Created on Thu May 23 13:54:48 2024

@author: HarisuShehu
"""

import os
import shutil
from PIL import Image

def copy_images_to_output(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List to store output folder paths
    output_folders = []

    # Iterate through each folder in the input folder
    for root, dirs, files in os.walk(input_folder):
        # Check if the folder name is a subject ID (e.g., S005)
        folder_name = os.path.basename(root)
        if folder_name.startswith('S') and folder_name[1:].isdigit():
            subject_id = int(folder_name[1:])

            # Create a corresponding output folder for the subject
            subject_output_folder = os.path.join(output_folder, str(len(output_folders) + 1))
            output_folders.append(subject_output_folder)
            if not os.path.exists(subject_output_folder):
                os.makedirs(subject_output_folder)

        # Copy images to the output directory
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith('.png') and Image.open(file_path).format == 'PNG':
                # Copy image to the next available subject folder
                subject_folder = output_folders[-1] if output_folders else None
                if subject_folder:
                    shutil.copy(file_path, subject_folder)


# Path to the input folder containing CK folder with images
input_folder = "./data/CK/"

# Path to the output folder where images will be copied
output_folder = "./data/Output"

# Copy images to the output folder
copy_images_to_output(input_folder, output_folder)
