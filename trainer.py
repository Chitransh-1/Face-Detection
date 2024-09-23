import os
import cv2
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
path = "dataset"

def get_images_with_id(path):
    
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    faces = []
    ids = []

    for image_path in image_paths:
        face_img = Image.open(image_path).convert('L')
        face_np = np.array(face_img, np.uint8)
        id = int(os.path.split(image_path)[-1].split(".")[1])
        print(f"ID: {id}, Image Path: {image_path}")

        faces.append(face_np)
        ids.append(id)
        cv2.imshow("Training", face_np)
        cv2.waitKey(10)

    return np.array(ids), faces

ids, faces = get_images_with_id(path)
recognizer.train(faces, ids)
recognizer.save("recognizer/trainingdata.yml")

cv2.destroyAllWindows()