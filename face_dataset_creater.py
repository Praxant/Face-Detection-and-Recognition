import cv2
import sqlite3

cam = cv2.VideoCapture(0)
detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def insertOrUpdate(id,name):
    conn = sqlite3.connect("FaceData.db")
    cmd = "SELECT * FROM Info WHERE ID= "+ str(id)
    cursor= conn.execute(cmd)
    isRecordExist=0
    for row in cursor:
        isRecordExist=1
    if(isRecordExist):
        cmd="UPDATE Info SET Name=' "+str(name)+" ' WHERE ID="+str(id)
    else:
        cmd="INSERT INTO Info(ID,Name) Values("+str(id)+",' "+str(name)+" ' )"
    conn.execute(cmd)
    conn.commit()
    conn.close()

Id =input('enter your id')
name = input('enter your name')
insertOrUpdate(Id,name)
sampleNum=0
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.8, 2)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        
        sampleNum=sampleNum+1
       
        cv2.imshow('frame',img)
        cv2.imwrite("dataSet/User."+str(Id) +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])

        
   
    cv2.waitKey(100)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
   
    elif sampleNum>20:
        break
cam.release()
cv2.destroyAllWindows()
print ("training....")
import trainner
