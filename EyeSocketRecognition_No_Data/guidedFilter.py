# -*- coding: utf-8 -*-
"""
Created on Thu May 23 16:46:13 2024

@author: HarisuShehu
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import cv2
import scipy.ndimage

# Define image dimensions
image_height = 128
image_width = 128
num_classes = 40

# Function to load images from 30 folders in a directory
def load_images_from_folders(parent_folder):
    images = []
    labels = []
    class_folders = sorted(os.listdir(parent_folder))
    for class_index, class_folder in enumerate(class_folders):
        class_path = os.path.join(parent_folder, class_folder)
        for filename in os.listdir(class_path):
            img = cv2.imread(os.path.join(class_path, filename))
            if img is not None:
                img = cv2.resize(img, (image_height, image_width))
                images.append(img)
                labels.append(class_index)  # Assigning label based on folder index
    return np.array(images), np.array(labels)

# Load data from folders and split into train, validation, and test sets
parent_folder = "./data/MaskedAT&T/"  # Path to the parent folder containing 30 class folders
images, labels = load_images_from_folders(parent_folder)
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.1, random_state=42)

# Implement guided filter manually
def guided_filter(I, p, radius=15, eps=1e-3):
    mean_I = scipy.ndimage.uniform_filter(I, size=(radius,radius,radius))
    mean_p = scipy.ndimage.uniform_filter(p, size=(radius,radius,radius))
    mean_Ip = scipy.ndimage.uniform_filter(I*p, size=(radius,radius,radius))
    cov_Ip = mean_Ip - mean_I*mean_p
    mean_II = scipy.ndimage.uniform_filter(I*I, size=(radius,radius,radius))
    var_I   = mean_II - mean_I*mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a*mean_I
    q = a*I + b
    return q

def apply_guided_filter(images, guide_images):
    processed_images = []
    for i in range(len(images)):
        processed_img = guided_filter(images[i], guide_images[i])
        processed_images.append(processed_img)
    return np.array(processed_images)

train_images_guided = apply_guided_filter(train_images, train_images)
val_images_guided = apply_guided_filter(val_images, val_images)
test_images_guided = apply_guided_filter(test_images, test_images)

# Define the model
model = models.Sequential()

# Convolutional layers
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(512, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Flatten layer
model.add(layers.Flatten())

# Dense layers
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))  # 30 classes for eye sockets

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

# Function to train and evaluate the model
def train_and_evaluate_model(train_images, train_labels, val_images, val_labels, test_images, test_labels):
    # Train the model
    history = model.fit(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels))

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_images, test_labels)

    return test_acc

# Perform analysis multiple times
num_runs = 30
results = []

for i in range(num_runs):
    print(f"Run {i+1}/{num_runs}")
    acc = train_and_evaluate_model(train_images_guided, train_labels, val_images_guided, val_labels, test_images_guided, test_labels)
    results.append(acc)

# Calculate mean and standard deviation
mean_accuracy = np.mean(results)
std_dev_accuracy = np.std(results)

# Print results
print(f"Mean accuracy: {mean_accuracy:.4f} +- {std_dev_accuracy:.4f}")


