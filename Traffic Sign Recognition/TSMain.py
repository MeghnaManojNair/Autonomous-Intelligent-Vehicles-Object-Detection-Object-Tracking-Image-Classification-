import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

#modelPath = '/home/mario/Graduation Project/Customize TSC/03-Classification/Models'
model = keras.models.load_model('TSModel5')

def returnRedness(img):
	yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
	y, u, v = cv2.split(yuv)
	return v

def threshold(img,T=150):
	_, img = cv2.threshold(img,T,255,cv2.THRESH_BINARY)
	return img

def findContour(img):
	contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	return contours

def findBiggestContour(contours):
	m = 0
	c = [cv2.contourArea(i) for i in contours]
	return contours[c.index(max(c))]

def boundaryBox(img,contours):
	x, y, w, h = cv2.boundingRect(contours)
	img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 1)
	sign = img[y:(y+h) , x:(x+w)]
	return img, sign

def preprocessingImageToClassifier(image=None,imageSize=28,mu=89.77428691773054,std=70.85156431910688):
    image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image,(imageSize,imageSize))
    image = (image - mu) / std
    image = image.reshape(1,imageSize,imageSize,1)
    return image

def predict(sign):
	img = preprocessingImageToClassifier(sign,imageSize=28)
	return np.argmax(model.predict(img))

#--------------------------------------------------------------------------
labelToText = { 0:"Stop",
    			1:"Do not Enter",
    			2:"Traffic Signal Ahead",
    			3:"Yeild - Give Way"}
cap=cv2.VideoCapture("video3.mp4")

while(True):
	_, frame = cap.read()
	redness = returnRedness(frame) # step 1 --> specify the redness of the image
	thresh = threshold(redness)
	try:
		contours = findContour(thresh)
		big = findBiggestContour(contours)

		if cv2.contourArea(big) > 3000:
			print(cv2.contourArea(big))
			img,sign = boundaryBox(frame,big)
			imresize = cv2.resize(img, (400, 400))
			cv2.imshow('frame',imresize)
			print("Detected Traffic Sign: ",labelToText[predict(sign)])
		else:
			cv2.imshow('frame',frame)

		'''
		if cv2.contourArea(big) > 3000:
			print(cv2.contourArea(big))
			img,sign = boundaryBox(frame,big)
			ims = cv2.resize(img, (400, 400))
			org = (50, 50)
			fontScale = 1
			color = (255, 0, 0)
			thickness = 2
			cv2.imshow('frame',ims)
			#cv2.putText(ims, "Hello", (210, 190),cv.FONT_HERSHEY_SIMPLEX, 1, (0, 25, 0), 3)

			print("Now,I see:",labelToText[predict(sign)])
			'''

	except:
		cv2.imshow('frame',frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
