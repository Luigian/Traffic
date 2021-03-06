import cv2
import numpy as np
import os
import sys
import tensorflow as tf

IMG_WIDTH = 30
IMG_HEIGHT = 30

# Check command-line arguments
if len(sys.argv) != 3:
    sys.exit("Usage: python recognition.py test_directory model")

test_directory = sys.argv[1]
model = tf.keras.models.load_model(sys.argv[2])

# Recognition of multiple images
for infile in os.listdir(test_directory):
    img = cv2.imread(os.path.join(test_directory, infile))
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    data = np.asarray(img)
    data = data / 255.0
    classification = model.predict([data.reshape(1, 30, 30, 3)]).argmax()
    name, ext = os.path.splitext(infile)
    print(f"{name}: {classification}")