# import the necessary packages
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

# grab the list of images that we'll be describing
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
# initialize the image preprocessor, load the dataset from disk,
# and reshape the data matrix
sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], 3072))
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

# Interactive section for testing a new image
if __name__ == "__main__":
    image_path = input("Enter the path to the image you want to test: ")
    predict_image(image_path, model, sp, le)
