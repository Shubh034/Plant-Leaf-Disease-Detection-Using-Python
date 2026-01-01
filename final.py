import cv2
import pandas as pd
import os
import csv 
import numpy as np
from skimage.feature import hog
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.utils import to_categorical
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import PIL
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

def extract_features(image):
  fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=True)
  return fd# Set parameters for HOG computation
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)

# Set directory paths
image_dir = '/content/drive/MyDrive/rice_leaf_diseases'
output_file = '/content/features.csv'

# Get list of subdirectories (assuming each subdirectory represents a class)
subdirs = sorted([os.path.join(image_dir, d) for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))])

# Initialize data and label arrays
data = []
labels = []

# Loop over subdirectories
for subdir in subdirs:
    print('Processing', subdir)
    # Get list of image files in the subdirectory
    files = sorted([os.path.join(subdir, f) for f in os.listdir(subdir) if os.path.isfile(os.path.join(subdir, f))])
    # Loop over images in the subdirectory
    for file in files:
        # Read image and resize it to a fixed size
        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (64, 64))
        # Compute HOG features and append to data array
        fd = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block)
        data.append(fd)
        # Extract label from directory name and append to label array
        label = subdir.split('/')[-1]
        labels.append(label)
#Here PCA needed to be perfromed as dataset very large number  of features 
features = np.array(data)
mean = np.mean(features, axis=0)
std = np.std(features, axis=0)
features = (features - mean) / std

# Perform PCA
pca = PCA(n_components=10)
pca.fit(features)
reduced_features = pca.transform(features)
labels = np.array(labels)
# Write data and labels arrays to CSV file
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['label'] + ['feature_{}'.format(i) for i in range(len(reduced_features[0]))])
    for i in range(len(data)):
        writer.writerow([labels[i]] + list(reduced_features[i]))

# Load the CSV file into a DataFrame
data = pd.read_csv('features.csv')
# Split the DataFrame into feature and target arrays
X = data.iloc[1:, 1:].values
y = data.iloc[1:, 0].values
print("Features-",X)
print("Labels-",y)
X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
print("X_train-",len(X_train))
print("X_test-",len(X_test))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# standardize feature values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# train logistic regression model
clf = LogisticRegression(random_state=0, max_iter=1000)
clf.fit(X_train, y_train)
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

# Calculate the training and testing accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Print the training and testing accuracy
print("Training Accuracy: {:.2f}%".format(train_accuracy*100))
print("Testing Accuracy: {:.2f}%".format(test_accuracy*100))
# predict on test set and evaluate model
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Split the dataset into training and testing sets
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# Preprocess the dataset
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train a Random Forest Classifier
classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

# Calculate the training and testing accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Print the training and testing accuracy
print("Training Accuracy: {:.2f}%".format(train_accuracy*100))
print("Testing Accuracy: {:.2f}%".format(test_accuracy*100))

# Evaluate the accuracy of the trained model
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Split the dataset into training and testing sets
X = data.iloc[1:, 1:].values
y = data.iloc[1:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Preprocess the dataset
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train a Naive Bayes Classifier
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

# Calculate the training and testing accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Print the training and testing accuracy
print("Training Accuracy: {:.2f}%".format(train_accuracy*100))
print("Testing Accuracy: {:.2f}%".format(test_accuracy*100))

# Evaluate the accuracy of the trained model
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Preprocess the dataset
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

# Calculate the training and testing accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Print the training and testing accuracy
print("Training Accuracy: {:.2f}%".format(train_accuracy*100))
print("Testing Accuracy: {:.2f}%".format(test_accuracy*100))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# Preprocess the dataset
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
clf = SVC()
clf.fit(X_train, y_train)
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

# Calculate the training and testing accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Print the training and testing accuracy
print("Training Accuracy: {:.2f}%".format(train_accuracy*100))
print("Testing Accuracy: {:.2f}%".format(test_accuracy*100))


import pathlib
image_dir = '/content/drive/MyDrive/rice_leaf_diseases'
data_dir=pathlib.Path(image_dir)
dict={"bacteria":list(data_dir.glob("1/*")),"brown":list(data_dir.glob("2/*")),"smut":list(data_dir.glob("3/*"))}
labels_dict = {
    'bacteria': 0,
    'brown': 1,
    'smut': 2,}
X, y = [], []
for name, images in dict.items():
    for image in images:
        img = cv2.imread(str(image))
        resized_img = cv2.resize(img,(180,180))
        X.append(resized_img)
        y.append(labels_dict[name])
X = np.array(X)
y = np.array(y)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=0)
len(X_test)
X_train_scaled = X_train / 255
X_test_scaled = X_test / 255
num_classes = 3
model = Sequential([
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
              
model.fit(X_train_scaled, y_train, epochs=30)


import matplotlib.pyplot as plt
from skimage import io
z = cv2.imread('/content/download.jpeg')
z = cv2.resize(z,(180,180))
z = z / 255.0
z = np.expand_dims(z, axis=0)
za = io.imread('/content/download.jpeg')
plt.imshow(za,cmap=plt.cm.gray)
pred = model.predict(z)
print(pred)
label = np.argmax(pred)
print('Labels with their values---[0] Bacterial Flight----[1] Brown Spot----[2] Leaf Smut')
print('Predicted label:', label)


        
