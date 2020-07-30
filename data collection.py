import numpy as np
import os
import cv2
import requests

x,y,w,h = 85, 200, 300, 200

cap = cv2.VideoCapture(0)

file_name = input("Enter the move: ")

i=0
ith = 163
start = False

url = "http://192.168.43.1:8080/shot.jpg"
while True:
	# ret, frame = cap.read()
	# if ret == False:
	# 	continue

	img_resp = requests.get(url)
	img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)

	frame = cv2.imdecode(img_arr, -1)

	gesture = frame[y:y+h,x:x+w]

	i+=1
	
	startkey = cv2.waitKey(1) & 0xFF
	if startkey == ord('a') and start == False:
		start = True
	
	if i%3 == 0 and start == True:
		ith+=1
		text = "Collected" + str(ith)
		cv2.putText(frame,text,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
		image_name = 'Collected Dataset/' + file_name +  '/' + str(ith) + '.jpg'
		cv2.imwrite(image_name, gesture)

	cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
	cv2.imshow("Frame", frame)

	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		break

print(file_name, "Data Saved Successfully")
