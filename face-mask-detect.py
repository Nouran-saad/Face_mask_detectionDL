# import all the needed packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

from imutils.video import VideoStream

import numpy as np
import imutils
import cv2


def detect_faces_and_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame 
	
	(h, w) = frame.shape[:2]
    # build the blob with the frame , no scaling ,resized image 224*224 and mean values for Red, Green, and Blue (RGB) channels
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network to make the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# initialize our list of faces, their  locations,
	# and the list of predictions from face mask network
	faces = []
	locations = []
	predictions = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence associated with the detection
		confidence = detections[0, 0, i, 2]


		# if detection greater than the minimum confidence keep it else filter it out
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box.
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to the lists
			faces.append(face)
			locations.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on all faces at the same time 

		faces = np.array(faces, dtype="float32")
		predictions = maskNet.predict(faces, batch_size=30)

	# return a 2-tuple of the face locations and their locations
	return (locations, predictions)

# load our face detector model
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model
maskNet = load_model("mask_detection.model")

# initialize the video stream
print("starting video stream...")
vs = VideoStream(src=0).start()

# read img by cv2 library
#img = cv2.imread("group.jpeg", cv2.IMREAD_COLOR)

# loop over every frames in the video stream
while True:
	# grab the frame from the  video stream and resize it to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# detect faces then determine if they are wearing a face mask or not
	(locations, predictions) = detect_faces_and_mask(frame, faceNet, maskNet)

	# loop over the detected face locations and the bounding box predictions 
	for (box, prediction) in zip(locations, predictions):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(mask, No_mask) = prediction

		# determine the class label and color the bounding box with label 'Mask' with blue and 'No Mask' with Red
		label = "Mask" if mask > No_mask else "No Mask"
		color = (255, 0, 0) if label == "Mask" else (0, 0, 255)

		# include the max probability between mask and No_mask in the label
		label = "{}: {:.2f}%".format(label, max(mask, No_mask) * 100)

		# display the label and bounding box rectangle on the result frame
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# cleanup
cv2.destroyAllWindows()
#vs.stop()
# To hold the window on screen, we use cv2.waitKey method
cv2.waitKey(0)