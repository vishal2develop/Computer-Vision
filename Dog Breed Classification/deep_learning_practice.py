# USAGE
# python deep_learning_with_opencv.py --image images/jemma.png --prototxt bvlc_googlenet.prototxt --model bvlc_googlenet.caffemodel --labels synset_words.txt

import numpy as np
import cv2
import time
import argparse

# Construct the commadn line arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i','--image', required=True, help="path to input image")
ap.add_argument('-p','--prototxt', required=True, help="Path to Caffe deploy prototxt file")
ap.add_argument('-m','--model', required=True, help="Path to the pre-trained output model")
ap.add_argument('-l','--labels', required=True, help="Path to imageNet labels(i.e,syn-nets)")
args = vars(ap.parse_args())

# --image : The path to the input image.
# --prototxt : The path to the Caffe “deploy” prototxt file.
# --model : The pre-trained Caffe model (i.e,. the network weights themselves).
# --labels : The path to ImageNet labels (i.e., “syn-sets”).

# Let’s load the input image and class labels:

# load the input image from disk
image = cv2.imread(args["image"])

# load the class labels from disk
rows = open(args["labels"]).read().strip().split("\n")
classes = [r[r.find(" ")+1:].split(",")[0] for r in rows]

# Now that we’ve taken care of the labels, let’s dig into the dnn  module

# our CNN requires fixed spatial dimensions for our input image(s)
# so we need to ensure it is resized to 224x224 pixels while
# performing mean subtraction (104, 117, 123) to normalize the input;
# after executing this command our "blob" now has the shape:
# (1, 3, 224, 224)

blob = cv2.dnn.blobFromImage(image,1,(224,224),(104,117,123))

# we use cv2.dnn.blobFromImage to perform mean subtraction to normalize the input image
# which results in a known blob shape

# We then load our model from disk:

# load our serialized model from disk
print("[INFO] loading model..")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# Now let’s complete a forward pass through the network with blob  as the input:

# set the blob as input to the network and perform a forward-pass to
# obtain our output classification

net.setInput(blob)
start = time.time()
preds = net.forward()
end = time.time()
print("[INFO] classification took {:.5} seconds".format(end-start))

# Let’s finish up by determining the top five predictions for our input image:

# sort the indexes of the probabilities in descending order (higher
# probabilitiy first) and grab the top-5 predictions

idxs = np.argsort(preds[0])[::-1][:5]

# display the top five class predictions:
# loop over the top-5 predictions and display them

for (i, idx) in enumerate(idxs):

    # draw the top prediction on the input image
    if i==0:
        text = "Label: {}, {:.2f}%".format(classes[idx], preds[0][idx] * 100)
        cv2.putText(image,text,(5,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    # display the predicted label + associated probability to the
    # console
    print("[INFO] {}. label: {}, probability: {:.5}".format(i+1, classes[idx], preds[0][idx]))

# display the output image
cv2.imshow("Image", image)
cv2.waitKey(0)

#breakdown:
# The idea for this loop is to
# (1) draw the top prediction label on the image itself and
# (2) print the associated class label probabilities to the terminal.

