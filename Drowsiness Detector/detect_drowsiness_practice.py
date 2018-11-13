import cv2
import imutils
from imutils import face_utils
import time
import argparse
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from threading import Thread
import numpy as np
import playsound
import dlib

# Thread  class so we can play our alarm in a separate thread from the main thread
# to ensure our script doesn’t pause execution while the alarm sounds.

# we need the playsound library, a pure Python, cross-platform implementation for playing simple sounds.


# we need to define our sound_alarm  function which accepts a path
# to an audio file residing on disk and then plays the file:

def sound_alarm(path):
    playsound.playsound(path)

# function which is used to compute the ratio of distances between the vertical eye landmarks
# and the distances between the horizontal eye landmarks:

def eye_aspect_ratio(eye):

    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A=dist.euclidean(eye[1],eye[5])
    B= dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates

    C= dist.euclidean(eye[0],eye[3])

    #compute the eye_aspect_ratio
    ear = (A+B) /(2.0 * C)

    #return ear

    return ear

# The return value of the eye aspect ratio will be approximately constant when the eye is open.
# The value will then rapid decrease towards zero during a blink.

# If the eye is closed, the eye aspect ratio will again remain approximately constant,
# but will be much smaller than the ratio when the eye is open.


# In the drowsiness detector case, we’ll be monitoring the eye aspect ratio to see
# if the value falls but does not increase again, thus implying that the person has closed their eyes.

# construct the argument parse and parse the arguments

ap = argparse.ArgumentParser()
ap.add_argument('-a','--alarm', type=str, default='', help='Path to the alarm file(.WAV) (Optional)')
ap.add_argument('-p','--shape-predictor', required=True, help='Path to the facial landmark detector')
ap.add_argument('-w','--webcam',type=int, default=0, help='index of webcam on system')
args = vars(ap.parse_args())


# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold for to set off the
# alarm

EYE_AR_THRESH = 0.3 #If the eye aspect ratio falls below this threshold,
# we’ll start counting the number of frames the person has closed their eyes for.
EYE_AR_CONSEC_FRAMES = 28 # If the number of frames the person has closed their eyes in exceeds this,
# we’ll sound an alarm.

# meaning that if a person has closed their eyes for 48 consecutive frames, we’ll play the alarm sound.

# initialize the frame counter as well as a boolean used to
# indicate if the alarm is going off
COUNTER = 0 # the total number of consecutive frames where the eye aspect ratio is below EYE_AR_THRESH .
# If COUNTER  exceeds EYE_AR_CONSEC_FRAMES , then we’ll update the boolean ALARM_ON
ALARM_ON = False

# Dlib part:

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args['shape_predictor'])

# To extract the eye regions from a set of facial landmarks,
# we simply need to know the correct array slice indexes:

# grab the indexes of the facial landmarks for the left and
# right eye, respectively

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

# Using these indexes, we’ll easily be able to extract the eye regions via an array slice.

# Core part

# start the video stream thread

print("[INFO] starting video stream thread...")
vs = VideoStream(src = args['webcam']).start()
time.sleep(1.0) #warmup time

# loop over frames from the video stream

while True:
    # grab the frame from the threaded video file stream, resize it and convert it to grayscale channels
    frame = vs.read()
    frame = imutils.resize(frame,width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray,0)

    # facial landmark detection to localize each of the important regions of the face:
    for rect in rects:

        # determine the facial landmarks for the face region,
        # then convert the facial landmark (x,y)-coordinates to a Numpy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates, then use the coordinates to compute the ear for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # average the ear together for both the eyes

        ear = (leftEAR + rightEAR) / 2.0

        # Visualization part:

        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame,[leftEyeHull], -1, (0,255,0),1)
        cv2.drawContours(frame,[rightEyeHull], -1, (0,255,0), 1)

        # Finally, we are now ready to check to see if the person in our video
        # stream is starting to show symptoms of drowsiness:

        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if ear <EYE_AR_THRESH:
            COUNTER+=1

        # if the eyes were closed for a sufficient number of time
        # then sound the alarm
            if COUNTER>=EYE_AR_CONSEC_FRAMES:


        # if alarm is not on , turn it on
                if not ALARM_ON:
                    ALARM_ON = True

            # check to see if an alarm file was supplied,
            # and if so, start a thread to have the alarm
            # sound played in the background
                    if args['alarm'] !='':
                        t = Thread(target=sound_alarm, args=(args['alarm'],))
                        t.daemon= True
                        t.start()


            # draw an alarm on the frame
                cv2.putText(frame, "Drowsiness Alert!", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # otherwise, the ear is not below the link threshold, so reset the counter and alarm
        else:
            COUNTER = 0
            ALARM_ON = False

            # The final code block in our drowsiness detector handles displaying the output frame  to our screen:

            # draw the computed eye aspect ratio on the frame to help
		    # with debugging and setting the correct eye aspect ratio
		    # thresholds and frame counters

        cv2.putText(frame, "EAR: {}".format(ear), (300,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)&0xFF

    # if q was pressed, break from loop
    if key==ord('q'):
        break

# cleanup
cv2.destroyAllWindows()
vs.stop()







