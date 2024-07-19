import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datasets import SimpleDatasetLoader
from preprocessing import SimplePreprocessor
from imutils import paths
import argparse
import numpy as np
import cv2
import os
import tkinter as tk
from tkinter import filedialog
import redis

# import the necessary packages
import matplotlib.pyplot as plt

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
    help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1,
    help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1,
    help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())

# check if the model exists in Redis
redis_host = 'localhost'
redis_port = 6379
redis_db = 1
redis_password = 'redis'

r = redis.Redis(host=redis_host, port=redis_port, db=redis_db, password=redis_password)
# check redis connected 
print(r.ping())
sp = SimplePreprocessor(32, 32)
if r.exists('data') and r.exists('labels'):
    # model exists in Redis, load it
    print("[INFO] Loading model from Redis...")
    data = r.get('data')
    data = pickle.loads(data)
    labels = r.get('labels')
    labels = pickle.loads(labels)
else:
    # model does not exist in Redis, train it
    # grab the list of images that we'll be describing
    print("[INFO] loading images...")
    imagePaths = list(paths.list_images(args["dataset"]))
    # initialize the image preprocessor, load the dataset from disk,
    # and reshape the data matrix
    
    sdl = SimpleDatasetLoader(preprocessors=[sp])
    (data, labels) = sdl.load(imagePaths, verbose=500)
    data = data.reshape((data.shape[0], 3072))
    r.set('data', pickle.dumps(data))
    r.set('labels', pickle.dumps(labels))
    
# show some information on memory consumption of the images
print("[INFO] features matrix: {:.1f}MB".format(
    data.nbytes / (1024 * 1024.0)))

# encode the labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)
# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# train and evaluate a k-NN classifier on the raw pixel intensities
print("[INFO] evaluating k-NN classifier...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"],
    n_jobs=args["jobs"])
model.fit(trainX, trainY)
print(classification_report(testY, model.predict(testX),
    target_names=le.classes_))

# Function to visualize an image and its prediction
def visualize_image(image, label, prediction):
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title(f"True Label: {label}\nPredicted: {prediction}")
    plt.axis("off")
    plt.show()

# Function to predict the label of a given image file
def predict_image(image_path, model, sp, le):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    image = sp.preprocess(image)
    image = image.flatten().reshape((1, -1))
    
    # Predict the label
    prediction = model.predict(image)[0]
    label = le.inverse_transform([prediction])[0]
    
    # Display the image and prediction
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Convert to RGB for plotting
    visualize_image(original_image, label, label)

# Function to open file dialog and predict image
def open_file_dialog():
    file_path = filedialog.askopenfilename()
    if file_path:
        predict_image(file_path, model, sp, le)

# Create the main window
root = tk.Tk()
root.title("Image Classifier")

# Create the browse button
browse_button = tk.Button(root, text="Browse Image", command=open_file_dialog)
browse_button.pack(pady=20)

# Start the GUI event loop
root.mainloop()
