#Importing required libraries
import os 
import numpy as np
import cv2
import facerecognition as fr

#C:\Users\VinodYedla9\Desktop\MYCODE\testimages
#Set the path for Traning and Testing images
test_img = cv2.imread(r'C:\Users\VinodYedla9\Documents\Major_Project\FaceRecognitionCode\testimages\9.jpg')

#Read a smaple trained image
faces_detected,gray_img= fr.faceDetection(test_img)
print("faces_detected",faces_detected)


faces,faceID=fr.labels_for_training_data('trainingimages')
face_recognizer=fr.train_classifier(faces,faceID)
face_recognizer.save('trainingData.yml')

#Assigning lables for the images
name={0:"Virat",1:"RajaReddy",2:"Rahman",3:"Vinod",4:"Venky",5:"Ramesh",6:"Obama",7:"Modi",8:"puttin",9:"SushmaSwaraj"}

#Opening Camera 
cap=cv2.VideoCapture(0)

while True:
	ret,test_img=cap.read()
	#Defining a function to detect faces of training images
	faces_detected,gray_img=fr.faceDetection(test_img)

	for (x,y,w,h) in faces_detected:
		cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)

	resized_img=cv2.resize(test_img, (1000,700))
	cv2.imshow('face detection ',resized_img)
	cv2.waitKey(10)	

	for face in faces_detected:
		(x,y,w,h)=face
		#Determines the coordinates of bounding box of detected face
		roi_gray=gray_img[y:y+h,x:x+h]
		label,confidence=face_recognizer.predict(roi_gray)
		print("confidence:",confidence)
		print("label",label)
		fr.draw_rect(test_img,face)
		#Calling the predict functions to predict the label of the test image
		predicted_name=name[label]
		if(confidence>67):
                        continue
		fr.put_text(test_img,predicted_name,x,y)
	resized_img=cv2.resize(test_img,(1000,700))
	cv2.imshow("face detection ",resized_img)
	if cv2.waitKey(10) == ord('q'):
		  break

cap.release()	
cv2.destroyAllWindows() 
