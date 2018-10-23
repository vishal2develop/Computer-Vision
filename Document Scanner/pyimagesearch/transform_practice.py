
import numpy as np
import cv2

def order_points(pts):

# initialzie a list of coordinates that will be ordered
# such that the first entry in the list is the top-left,
# the second entry is the top-right, the third is the
# bottom-right, and the fourth is the bottom-left

    rect=np.zeros((4,2),dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1) # column wise sum
    rect[0]=pts[np.argmin(s)]
    rect[2]=pts[np.argmax(s)]

    # now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
    diff = np.diff(pts,axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered co-ordinates
    return rect

#function breakdown:
#  order_points  function on Line 5. This function takes a single argument, pts ,
# which is a list of four points specifying the (x, y) coordinates of each point of the rectangle.

# It is absolutely crucial that we have a consistent ordering of the points in the rectangle.
# The actual ordering itself can be arbitrary, as long as it is consistent throughout the implementation.

# We’ll start by allocating memory for the four ordered points using np.zeros().

# Again, I can’t stress again how important it is to maintain a consistent ordering of points.
# You will se why.

def four_point_transform(image,pts):
    # obtain a consistent order of the points and unpack them individually
    rect=order_points(pts)
    (tl,tr,br,bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) **2) + ((br[1] - bl[1]) **2))
    widthB = np.sqrt(((tr[0] - tl[0]) **2) + ((tr[1] - tl[1]) **2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the maximum distance between the top-right and
    #  bottom-right y-coordinates or the top-left and bottom-left y-coordinates

    heightA = np.sqrt(((tr[0] - br[0]) **2) + ((tr[1] - br[1]) **2))
    heightB = np.sqrt(((tl[0] - bl[0])**2) + ((tl[1] - bl[1]) **2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left order
    dst = np.array([
        [0,0],
        [maxWidth-1,0],
        [maxWidth-1,maxHeight-1],
        [0,maxHeight-1]
    ], dtype="float32")

    # You can see why. Here, we define 4 points representing our “top-down” view of the image.
    # The first entry in the list is (0, 0)  indicating the top-left corner.
    # The second entry is (maxWidth - 1, 0)  which corresponds to the top-right corner.
    # Then we have (maxWidth - 1, maxHeight - 1)  which is the bottom-right corner.
    # Finally, we have (0, maxHeight - 1)  which is the bottom-left corner.

    # compute the perspective transform matrix and then apply it

    M = cv2.getPerspectiveTransform(rect,dst)
    warped = cv2.warpPerspective(image,M,(maxWidth,maxHeight))

    # To actually obtain the top-down, “birds eye view” of the image we’ll utilize the cv2.getPerspectiveTransform
    # rect - list of 4 ROI points in the original image,
    # dst - list of transformed points.
    # The cv2.getPerspectiveTransform  function returns M , which is the actual transformation matrix.

    # The output of cv2.warpPerspective  is our warped  image, which is our top-down view.

    # return the warped image
    return warped



# Now that we have code to perform the transformation, we need some code to drive it and actually apply it to images.




