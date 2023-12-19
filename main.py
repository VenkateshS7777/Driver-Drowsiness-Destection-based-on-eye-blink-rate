from scipy.spatial import distance
from imutils import face_utils
import numpy as np
from pygame import mixer
import imutils
import dlib
import cv2
import webbrowser,time

mixer.init()
mixer.music.load("alarm.wav")

def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear
	
count = 0
total = 0
thresh = 0.25
frame_check = 20
frame_check_1 = 30
YAWN_THRESH = 20
ear=0
distance1=0

#detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

cap=cv2.VideoCapture(0)

flag=0
while True:
	ret, frame=cap.read()
	frame = imutils.resize(frame, width=550)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	subjects = detect(gray, 0)
	for subject in subjects:
		shape = predict(gray, subject)
		shape = face_utils.shape_to_np(shape)
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		ear = (leftEAR + rightEAR) / 2.0        
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
	def lip_distance1(shape):
		top_lip = shape[50:53]
		top_lip = np.concatenate((top_lip, shape[61:64]))

		low_lip = shape[56:59]
		low_lip = np.concatenate((low_lip, shape[65:68]))

		top_mean = np.mean(top_lip, axis=0)
		low_mean = np.mean(low_lip, axis=0)
		
		distance1 = abs(top_mean[1] - low_mean[1])
		return distance1	
	
	faces=detect(gray)
	
	for face in faces:
		x,y=face.left(),face.top()
		hi,wi=face.right(),face.bottom()
		cv2.rectangle(frame,(x,y),(hi,wi),(0,0,255),2)
		cv2.putText(frame,'Face Detected',(x-12,y-12),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
		distance1 = lip_distance1(shape)
		lip = shape[48:60]
		cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

			
		if ear < thresh:
			flag += 1
			print (flag)
			if flag >= frame_check:
				cv2.putText(frame, "*********!!!DROWSINESS ALERT!!!*********", (10,20),
					cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
				cv2.putText(frame, "****!!!PLEASE WAKEUP IMMEDIATELY!!!****", (10,370),
					cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 370), 2)
				mixer.music.play()
		else:
			flag=0

		if(flag>=frame_check_1):
			webbrowser.open("https://www.google.com/maps/search/hotels+or+motels+near+me")
			time.sleep(5)
				
		if (distance1 > YAWN_THRESH):
			cv2.putText(frame, "!!!YAWN ALERT!!!", (210, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			mixer.music.play()
			
		if ear>0.3:
			count+=1
		else:
			if count>=3:
				total+=1
				count=0
	cv2.putText(frame,"BLINK COUNT: {}".format(total), (10, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	cv2.putText(frame, "EAR: {:.2f}".format(ear), (10, 100),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	cv2.putText(frame, "YAWN: {:.2f}".format(distance1), (400, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)               
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
cv2.destroyAllWindows()
cap.release() 