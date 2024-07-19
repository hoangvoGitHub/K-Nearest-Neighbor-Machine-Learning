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

# Function to visualize some of the test images and their predictions
def visualize_predictions(testX, testY, predictions, imagePaths, le, num_images=10):
    plt.figure(figsize=(10, 10))
    indices = np.random.choice(len(testX), num_images, replace=False)
    
    for i, idx in enumerate(indices):
        image = cv2.imread(imagePaths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for plotting
        label = le.classes_[testY[idx]]
        pred = le.classes_[predictions[idx]]
        
        plt.subplot(5, 5, i + 1)
        plt.imshow(image)
        plt.title(f"True: {label}\nPred: {pred}")
        plt.axis("off")
    
    plt.show()

# Predict on the test set
predictions = model.predict(testX)

# Visualize predictions
print("[INFO] visualizing predictions...")
visualize_predictions(testX, testY, predictions, imagePaths, le)
