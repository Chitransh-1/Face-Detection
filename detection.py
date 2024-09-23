import cv2
import sqlite3

face_detection = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recognizer/trainingdata.yml")

def getProfile(id):
    conn = sqlite3.connect("sqlite.db")
    cursor = conn.execute("SELECT * FROM students WHERE id = ?", (id,))
    profile = None
    for row in cursor:
        profile = row
        print("Roww", profile)
    conn.close()
    return profile

while(True):
    ret, img = video_capture.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray, 1.3, 5)

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
        id, conf = recognizer.predict(gray[y: y+h, x: x+w])
        profile = getProfile(id)
        #print(profile)

        if(profile!= None):
            cv2.putText(img, "Roll No. : " +str(profile[0]), (x, y+h+75), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 129), 2)
            cv2.putText(img, "Name : " +str(profile[1]), (x, y+h+20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 129), 2)
            cv2.putText(img, "Age : " +str(profile[2]), (x, y+h+45), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 129), 2)
    
    cv2.imshow("FACE", img)
    if(cv2.waitKey(1)==ord('q')):
        break

video_capture.release()
cv2.destroyAllWindows()