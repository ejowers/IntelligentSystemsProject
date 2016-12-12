# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Dense
from keras.utils import np_utils
from fuzzywuzzy import fuzz
from imutils import paths
import numpy as np
import argparse
import cv2
import os

# -------------------------------- FUNCTIONS --------------------------------- #

# resize the image to a fixed size, then flatten the image into a list of raw pixel intensities
def image_to_feature_vector(image, size=(32, 32)):
	return cv2.resize(image, size).flatten()

# ----------------------------- TRAINING IMAGES ------------------------------ #

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--training_set", required=True, help="path to training dataset")
args = vars(ap.parse_args())

# grab the list of images that we'll be describing
print("[INFO] describing images...")
imagePaths = list(paths.list_images(args["training_set"]))


# initialize the data matrix and labels list
data = []
labels = []

# loop over the input images
for (i, imagePath) in enumerate(imagePaths):
	# load the image and extract the class label (assuming that our
	# path as the format: /path/to/dataset/{class}.{image_num}.jpg
	image = cv2.imread(imagePath)
	label = imagePath.split(os.path.sep)[-1].split(".")[0]

	# construct a feature vector raw pixel intensities, then update
	# the data matrix and labels list
	features = image_to_feature_vector(image)
	data.append(features)
	labels.append(label)

	# show an update every 1,000 images
	if i > 0 and i % 1000 == 0:
		print("[INFO] processed {}/{}".format(i, len(imagePaths)))

# encode the labels, converting them from strings to integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# scale the input image pixels to the range [0, 1], then transform
# the labels into vectors in the range [0, num_classes] -- this
# generates a vector for each label where the index of the label
# is set to `1` and all other entries to `0`
data = np.array(data) / 255.0
labels = np_utils.to_categorical(labels, 26)

# -------------------------------- DICTIONARY -------------------------------- #

# open dictionary file
f = open('dictionary.txt', 'r')

# convert all words to uppercase
dictionary = []
for line in f:
	dictionary += [line.upper()]

# ------------------------------ NEURAL NETWORK ------------------------------ #

# partition the data into training and testing splits, using 75%
# of the data for training and the remaining 25% for testing
print("[INFO] constructing training/testing split...")
(trainData, testData, trainLabels, testLabels) = train_test_split(
	data, labels, test_size=0.25, random_state=42)

# define the architecture of the network
model = Sequential()
model.add(Dense(768, input_dim=3072, init="uniform",
	activation="relu"))
model.add(Dense(384, init="uniform", activation="relu"))
model.add(Dense(26))
model.add(Activation("softmax"))

# train the model using SGD
print("[INFO] compiling model...")
sgd = SGD(lr=0.01)
model.compile(loss="categorical_crossentropy", optimizer=sgd,
	metrics=["accuracy"])
model.fit(trainData, trainLabels, nb_epoch=500, batch_size=78,
	verbose=1)

# show the accuracy on the testing set
print("\n[INFO] valuating on testing set...")
(loss, accuracy) = model.evaluate(testData, testLabels,
	batch_size=78, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,
	accuracy * 100))

# ------------------------------ TRANSLATE WORD ------------------------------ #

# loop through different words without retraining
test_folder = "test"
while test_folder[0] == 't':

	# takes input from the user for which test to run
	test_folder = raw_input('\nEnter the test to run: ')

	# Exits the loop when type 
	if test_folder[0] != 't':
		break

	# grab the list of images that we'll be translating
	word = list(paths.list_images(test_folder))
	word.sort()

	# loop over the input images
	wdata = []
	for (i, letter) in enumerate(word):
		# load the image and extract the class label (assuming that our
		# path as the format: /path/to/dataset/{class}.jpg
		image = cv2.imread(letter)

		# construct a feature vector raw pixel intensities, then update
		# the data matrix and labels list
		features = image_to_feature_vector(image)
		wdata.append(features)

	wdata = np.array(wdata) / 255.0

	result = model.predict(wdata).round()

	# make result into a word
	word_result = ""
	ascii_value = 0
	for letter in result:
		letter = letter.tolist()
		ascii_value = letter.index(1) + 65
		word_result += chr(ascii_value)

	print "\nThe neural network spit out: %s" % word_result
	# take in dictionary at beginning
	# change all words to lowercase
	# run for loop through dictionary checking for highest match
	best_match = ""
	best_percent = 0
	size = len(dictionary)
	for word in dictionary:
		percent = fuzz.ratio(word_result, word)
		if percent > best_percent:
				best_percent = percent
				best_match = word

	print "\nThe result is %swith %d%% certainty." % (best_match, best_percent)

print "Program Exitting..."
