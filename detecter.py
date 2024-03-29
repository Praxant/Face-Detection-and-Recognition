import cv2
import numpy as np
import sqlite3

recog = cv2.face.LBPHFaceRecognizer_create()
recog.read('trainningData.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceDetect = cv2.CascadeClassifier(cascadePath);


def getProfile(id):
	conn=sqlite3.connect("FaceData.db")
	cmd = "SELECT * FROM Info WHERE ID ="+ str(id)
	cursor = conn.execute(cmd)
	profile = None
	for row in cursor:
		profile = row
	conn.close()
	return profile



cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret, img =cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray, 1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        ID, conf = recog.predict(gray[y:y+h,x:x+w])
        if(conf>40):
            profile = getProfile(ID)
        else:
        	ID=0
        	profile= getProfile(ID)

        if(profile!=None):
        	cv2.putText(img,str(profile[1]),(x,y+h+30),font,0.55,(0,255,0),1)
        	cv2.putText(img,str(profile[0]),(x,y+h+60),font,0.55,(0,255,0),1)
        	cv2.putText(img,str(conf),(x,y+h+90),font,0.55,(0,255,0),1)
        # elif(profile==None):
        # 	cv2.putText(img,"Unknown",(x,y+h+30),font,0.55,(0,255,0),1)
        # 	cv2.putText(img,"Unknown",(x,y+h+60),font,0.55,(0,255,0),1)

    cv2.imshow('Face',img) 
    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
