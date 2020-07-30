import requests
import numpy as np
import cv2
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
import random

# print(tf.__version__)

model_filepath = 'my_model.h5'

model = tf.keras.models.load_model(
    model_filepath,
    custom_objects=None,
    compile=False
)

url = "http://192.168.43.1:8080/shot.jpg"

label2gesture = {
    0:'Rock',
    1:'Paper',
    2:'Scissor',
    3:'None'
}
	
x,y,w,h = 10, 200, 300, 200

cap = cv2.VideoCapture(0)


rock = 'Collected Dataset/Rock/1.jpg'
paper = 'Collected Dataset/Paper/1.jpg'
scissor = 'Collected Dataset/Scissor/1.jpg'

def getWinner(user_move, comupter_move):

	if user_move == "None" or comupter_move == "None":
		return "none"
	if user_move == "Rock":
		if comupter_move == "Scissor":
			return "user"
		if comupter_move == "Paper":
			return "computer"
	
	if user_move == "Paper":
		if comupter_move == "Scissor":
			return "computer"
		if comupter_move == "Rock":
			return "user"
	
	if user_move == "Scissor":
		if comupter_move == "Rock":
			return "computer"
		if comupter_move == "Paper":
			return "user"	


while True:

	# User Move.
	img_resp = requests.get(url)
	img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)

	img = cv2.imdecode(img_arr, -1)

	cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
	
	img_section = img[y:y+h,x:x+w]
	
	test = image.img_to_array(img_section)/255.0
	test = test.reshape(1,*test.shape)

	pred = model.predict(test)

	user_move = label2gesture[np.argmax(pred, axis=1)[0]]

	cv2.putText(img,"User: ",(x,y-40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
	cv2.putText(img,user_move,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)


	# Computer's Move.
	move = random.randint(0,2)
	comupter_move = label2gesture[move]

	cimg = ""
	if comupter_move == "Rock":
		cimg = 'Collected Dataset/Rock/1.jpg'
	if comupter_move == "Paper":
		cimg = 'Collected Dataset/Paper/1.jpg'
	if comupter_move == "Scissor":
		cimg = 'Collected Dataset/Scissor/1.jpg'

	cv2.rectangle(img,(x+310,y),(x+310+w,y+h),(0,255,255),2)
	cv2.putText(img,"Computer: ",(x+310,y-40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
	cv2.putText(img,comupter_move,(x+310,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
			
	winner = getWinner(user_move, comupter_move)

	cv2.putText(img,"Winner: " + str(winner),(x+150,y-100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3,cv2.LINE_AA)

	c_image = image.load_img(cimg)
	img_array = image.img_to_array(c_image)
	img[y:y+h,x+310:x+310+w] = img_array

	cv2.imshow("AndroidCam", img)

	cv2.waitKey(3000)
	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		break
