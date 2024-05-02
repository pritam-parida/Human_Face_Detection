import tensorflow as tf
from tensorflow.keras import layers, models
import os
import cv2
import numpy as np

# Define a simple CNN model for face detection
def create_face_detection_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Output layer (binary classification)

    return model

# Load and preprocess images from a directory
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):  # Check if it's a file (ignore directories)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (128, 128))  # Resize to 128x128
                img = img / 255.0  # Normalize pixel values (0-1)
                images.append(img)
    return np.array(images)

# Load face and non-face images
faces = load_images_from_folder('images')
non_faces = load_images_from_folder('non')

# Create labels (1 for face, 0 for non-face)
face_labels = np.ones(len(faces))
non_face_labels = np.zeros(len(non_faces))

# Concatenate images and labels
X = np.concatenate((faces, non_faces), axis=0)
y = np.concatenate((face_labels, non_face_labels), axis=0)

# Shuffle the data
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# Create and compile the model
model = create_face_detection_model()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Set window size and position for display
cv2.namedWindow('Face Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Face Detection', 800, 600)  # Set desired window size

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame from the webcam.")
        break

    # Preprocess the frame (resize, normalize, expand dims)
    frame_resized = cv2.resize(frame, (128, 128))
    frame_resized = frame_resized / 255.0  # Normalize pixel values (0-1)
    frame_input = np.expand_dims(frame_resized, axis=0)  # Add batch dimension

    # Use the model to predict if the frame contains a face
    prediction = model.predict(frame_input)

    # Determine prediction label and display text
    if prediction[0][0] >= 0.5:
        text = "Face Detected"
        color = (0, 255, 0)  # Green color for positive detection
    else:
        text = "No Face Detected"
        color = (0, 0, 255)  # Red color for negative detection

    # Draw text on the frame
    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Display the frame
    cv2.imshow('Face Detection', frame)

    # Wait for key press to exit (press 'q' to quit)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
