import cv2,os
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create();
path="dataSet"

detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def getImageWithID(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    IDs=[]
    for imagePath in imagePaths:
        faceImg=Image.open(imagePath).convert('L');
        faceNp=np.array(faceImg,'uint8')
        ID=int(os.path.split(imagePath)[-1].split('.')[1])

        facecascades= detector.detectMultiScale(faceNp)
        for (x,y,w,h) in facecascades:
            faces.append(faceNp[y:y+h,x:x+w])
            IDs.append(ID)

        # faces.append(faceNp)
        # IDs.append(ID)
        # cv2.imshow('trainning',faceNp)
        # cv2.waitKey(10)
      
    return IDs,faces

Ids,faces=getImageWithID(path)
recognizer.train(faces,np.array(Ids))
recognizer.write('trainningData.yml')
cv2.destroyAllWindows()

