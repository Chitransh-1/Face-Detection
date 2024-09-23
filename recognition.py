import cv2
import sqlite3

face_detect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)

def insert_update(id, name, age):
    conn = sqlite3.connect("sqlite.db")
    myquery = "SELECT * FROM students WHERE ID = ?"
    mycursor = conn.execute(myquery, (id,))
    isRecordExist = 0

    for row in mycursor:
        isRecordExist = 1

    if isRecordExist == 1:
        conn.execute("UPDATE students SET name = ?, age = ? WHERE id = ?", (name, age, id))
    else:
        conn.execute("INSERT INTO students (id, name, age) VALUES (?, ?, ?)", (id, name, age))

    conn.commit()
    conn.close()

id = input("Enter user id : ")
name = input("Enter name : ")
age = input("Enter age : ")

insert_update(id, name, age)

sampleNum = 0

while True:
    ret, img = video_capture.read()
    if not ret:
        print("Failed to capture image.")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        sampleNum += 1
        cv2.imwrite("dataset/user." + str(id) + "." + str(sampleNum) + ".jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.waitKey(100)
    
    cv2.imshow("Face", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if sampleNum > 20:
        break

video_capture.release()
cv2.destroyAllWindows()