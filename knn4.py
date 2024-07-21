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
from PIL import Image
import requests
# import the necessary packages
import matplotlib.pyplot as plt
from io import BytesIO
from tkinter import simpledialog
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
print('Connected redis: ',r.ping())
if r.exists('knn-data') and r.exists('knn-labels'):
    # model exists in Redis, load it
    print("[INFO] Loading model from Redis...")
    data = r.get('knn-data')
    data = pickle.loads(data)
    labels = r.get('knn-labels')
    labels = pickle.loads(labels)
else:
    # model does not exist in Redis, train it
    # grab the list of images that we'll be describing
    print("[INFO] loading images...")
    imagePaths = list(paths.list_images(args["dataset"]))
    # initialize the image preprocessor, load the dataset from disk,
    # and reshape the data matrix
    
    sdl = SimpleDatasetLoader(preprocessors=[sp])
    (data, labels) = sdl.load(imagePaths, verbose=500, r=r)
    data = data.reshape((data.shape[0], 3072))
    r.set('knn-data', pickle.dumps(data))
    r.set('knn-labels', pickle.dumps(labels))
sp = SimplePreprocessor(32, 32)
    
# show some information on memory consumption of the images
print("[INFO] features matrix: {:.1f}MB".format(
    data.nbytes / (1024 * 1024.0)))

# encode the labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)
# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(X_train, X_test, y_train, y_test) = train_test_split(data, labels, test_size=0.25, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_test, y_test, random_state=42, test_size=0.5)

print("X_train: "+str(X_train.shape))
print("X_test: "+str(X_test.shape))
print("X_val: "+str(X_val.shape))
print("y_train: "+str(y_train.shape))
print("y_test: "+str(y_test.shape))
print("y_val: "+str(y_val.shape)+'\n')

num_training = X_train.shape[0]
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]

num_test = X_test.shape[0]
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]

num_val = X_val.shape[0]
mask = list(range(num_val))
X_val = X_val[mask]
y_val = y_val[mask]

# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))

print("X_train: "+str(X_train.shape))
print("X_test: "+str(X_test.shape))
print("X_val: "+str(X_val.shape))
print("y_train: "+str(y_train.shape))
print("y_test: "+str(y_test.shape))
print("y_val: "+str(y_val.shape))

num_val = X_val.shape[0]
mask = list(range(num_val))

print("Using SKLEARN")
if not r.exists('knn-scores'):
    lix = []
    liy = []
    bestK = 0
    acc = 0
    for k in range(1, 100):
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(X_train, y_train)
        score = neigh.score(X_val, y_val)
        liy.append(score)
        if score > acc:
            acc = score
            bestK = k - 1
        lix.append(k)
    
    # Store the scores in Redis
    r.set('knn-scores', pickle.dumps((lix, liy, bestK, acc)))
else:
    # Retrieve the scores from Redis
    knn_scores = pickle.loads(r.get('knn-scores'))
    lix, liy, bestK, acc = knn_scores

plt.plot(lix, liy)
plt.show()
print("max acc at k="+str(bestK+1)+" acc of "+str(acc))

model = KNeighborsClassifier(bestK+1)
model.fit(X_train, y_train)
# Function to visualize an image and its prediction
def visualize_image(original_image, closest_images, label, prediction):
    fig, axs = plt.subplots((len(closest_images) + 1) // 2, 2, figsize=(12, 6))
    axs = axs.flatten()

    axs[0].imshow(original_image)
    axs[0].set_title(f"True Label: {label}\nPredicted: {prediction}")
    axs[0].axis("off")

    for i, image in enumerate(closest_images):
        closest_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axs[i + 1].imshow(closest_image)
        axs[i + 1].set_title(f"Closest Training Image {i+1}")
        axs[i + 1].axis("off")

    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()

# Function to predict the label of a given image file
def predict_image(image_path, model, sp, le, image_size, X_train):
    image = cv2.imread(image_path)
    if image is None:
        print(f"[WARNING] Unable to load image at path: {image_path}")
        return
    image = cv2.resize(image, image_size)
    image = sp.preprocess(image)
    image = image.flatten().reshape((1, -1))
    
    prediction = model.predict(image)[0]
    label = le.inverse_transform([prediction])[0]
    
    distances, indices = model.kneighbors(image, n_neighbors=bestK + 1)
    closest_images = [X_train[index].reshape(32, 32, 3) for index in indices[0]]
    
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    visualize_image(original_image, closest_images, label, label)

# Function to predict the label of a given image URL
def predict_image_from_url(image_url, model, sp, le, image_size, X_train, trainImages):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert('RGB')
    image = np.array(image)
    
    image = cv2.resize(image, image_size)
    image = sp.preprocess(image)
    image = image.flatten().reshape((1, -1))
    
    prediction = model.predict(image)[0]
    label = le.inverse_transform([prediction])[0]
    
    distances, indices = model.kneighbors(image, n_neighbors=bestK + 1)
    closest_images = [X_train[index].reshape(32, 32, 3) for index in indices[0]]
    
    original_image = np.array(Image.open(BytesIO(response.content)).convert('RGB'))
    visualize_image(original_image, closest_images, label, label)

# Function to open file dialog and predict image
def open_file_dialog():
    file_path = filedialog.askopenfilename()
    if file_path:
        predict_image(file_path, model, sp, le, (32, 32), X_train)

# Function to open input dialog for URL and predict image
def open_url_dialog():
    image_url = simpledialog.askstring("Input", "Enter image URL:")
    if image_url:
        predict_image_from_url(image_url, model, sp, le, (32, 32), X_train, trainImages)

trainImages = pickle.loads(r.get('data'))

# Create the main window
root = tk.Tk()
root.title("Image Classifier")

# Create the browse button
browse_button = tk.Button(root, text="Browse Image", command=open_file_dialog)
browse_button.pack(pady=20)

# Create the URL input button
url_button = tk.Button(root, text="Predict Image from URL", command=open_url_dialog)
url_button.pack(pady=20)
# Start the GUI event loop
root.mainloop()
