import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.utils import image_dataset_from_directory
import os
import random
import cv2

# Function to visualize images
def visualize_images(path, num_images=5):
    image_filenames = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    if not image_filenames:
        raise ValueError("No images found in the specified path")
    selected_images = random.sample(image_filenames, min(num_images, len(image_filenames)))
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3), facecolor='white')
    for i, image_filename in enumerate(selected_images):
        image_path = os.path.join(path, image_filename)
        image = plt.imread(image_path)
        axes[i].imshow(image)
        axes[i].axis('off')
        axes[i].set_title(image_filename)
    plt.tight_layout()
    plt.show()

# Load datasets
train_ds = keras.utils.image_dataset_from_directory(
    directory=r'C://path_to_extract/Training',
    labels='inferred',
    label_mode='int',
    batch_size=64,
    image_size=(256, 256)
)

val_ds = keras.utils.image_dataset_from_directory(
    directory=r'C://path_to_extract/Testing',
    labels='inferred',
    label_mode='int',
    batch_size=64,
    image_size=(256, 256)
)

classes = train_ds.class_names

# Visualize images
for label in classes:
    path_to_visualize = f"/content/path_to_extract/Training/{label}"
    print(label.upper())
    visualize_images(path_to_visualize, num_images=4)

# Build and compile CNN model
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_ds, epochs=10, validation_data=val_ds, verbose=1)

# Plot training history
plt.title('Training Accuracy and Validation Accuracy')
plt.plot(history.history['accuracy'], color='red', label='Training')
plt.plot(history.history['val_accuracy'], color='blue', label='Validation')
plt.legend()
plt.show()

plt.title('Training Loss and Validation Loss')
plt.plot(history.history['loss'], color='red', label='Training')
plt.plot(history.history['val_loss'], color='blue', label='Validation')
plt.legend()
plt.show()

# Save the trained classification model
model.save('brain_tumor_prediction.h5')

# Load YOLO model for object detection
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Function to detect objects in an image
def detect_objects(image_path):
    img = cv2.imread(image_path)
    height, width, channels = img.shape

    # Create a blob from the image
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialize lists for detected bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Adjust confidence threshold as needed
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = (0, 255, 0)  # Green for bounding boxes
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, f'{label} {confidence}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img

# Path to your image
image_path = 'path_to_your_image.jpg'

# Detect objects in the image
marked_image = detect_objects(image_path)

# Display the result
plt.imshow(cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()