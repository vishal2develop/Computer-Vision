 # Whenever you need to perform a 4 point perspective transform, you should be using the transform.py module
# we’ll be using it to build our very own document scanner.

# importing the necessary packages
import cv2
import imutils
import argparse
import numpy as np
from pyimagesearch.transform_practice import four_point_transform
from skimage.filters import threshold_local

# import the threshold_local function from scikit-image.
# This function will help us obtain the “black and white” feel to our scanned image.

# construct the command line arguments
ap = argparse.ArgumentParser()
ap.add_argument('--image','-i', required=True, help="path to the image")
args = vars(ap.parse_args())

# Step 1: Edge Detection

# load the image and compute the ratio of the old height
# to the new height, clone it, and resize it

img = cv2.imread(args['image'])
ratio = img.shape[0]/500.0
orig = img.copy()
img = imutils.resize(img,height=500)

# In order to speedup image processing, as well as make our edge detection step more accurate,
# we resize our scanned image to have a height of 500 pixels

# We also take special care to keep track of the ratio  of the original height of the image to the new height
# this will allow us to perform the scan on the original image rather than the resized image.

# convert the image to grayscale, blur it, and find edges in the image

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
edged = cv2.Canny(blur, 75, 200)

# show the original image and the edge detected image
print("Step 1: Edge Detection")
cv2.imshow("Image", img)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 2: Finding contours

# The trick here is to assume that the largest contour in the image with exactly four points
#  is our piece of paper to be scanned
# This is a safe assumption as we are building a document scanner and it scans a piece of paper.
# it will also be the largest object in the image as when clicking the picture the focus will be on the document
# Therefore, we can discard other contours and keep only the largest contours

# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour

cnts = cv2.findContours(edged.copy(),cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

# We start off by finding the contours in our edged image .
# We also handle the fact that OpenCV 2.4 and OpenCV 3 return contours differently.
# Then we sort the contours by area and keep only the largest ones.
# This allows us to only examine the largest of the contours, discarding the rest.

# Loop over the contours
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c,0.02 * peri, True)

    # if our approximated contour has four points, then we
    # can assume that we have found our screen
    if len(approx) ==4:
        screenCnt = approx
        break

# show the contour (outline) of the piece of paper
print("Step2: Find Contours of paper")
cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline",img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# If the approximated contour has four points (Line 49), we assume that we have found the document in the image.
# And again, this is a fairly safe assumption. The scanner app will assume that
# (1) The document to be scanned is the main focus of the image and
# (2) the document is rectangular, and thus will have four distinct edges.


# Step 3: Apply a Perspective Transform & Threshold

# The last step in building a mobile document scanner is to take the four points representing the
# outline of the document and apply a perspective transform to obtain a top-down, “birds eye view” of the image.

# Use the transform.py module we created earlier

# apply the four point transform to obtain a top-down view of the original image
warped = four_point_transform(orig, screenCnt.reshape(4, 2)*ratio)

# convert the warped image to grayscale, then threshold it
# to give it that 'black and white' paper effect

warped = cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset=10, method='gaussian')
warped = (warped > T).astype('uint8')*255


# show the original and scanned images

print("Step3: Apply Perspective Transform")
cv2.imshow("Original", imutils.resize(orig, height=650))
cv2.imshow("Scanned", imutils.resize(warped, height=650))
cv2.waitKey(0)




# We’ll pass two arguments into four_point_transform:
# img - original image (not the resized one)
# the second argument is the contour representing the document, multiplied by the resized ratio.

# We multiply by the resized ratio because we performed edge detection
# and found contours on the resized image of height=500 pixels.

# However, we want to perform the scan on the original image, not the resized image,
# thus we multiply the contour points by the resized ratio.

# To obtain the black and white feel to the image, we then take the warped image,
#  convert it to grayscale and apply adaptive thresholding










