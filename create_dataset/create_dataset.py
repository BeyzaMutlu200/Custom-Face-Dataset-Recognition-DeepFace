import cv2
import sys
import os

face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def face_crop(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return None

    for (x, y, w, h) in faces:
        face_crop = img[y:y+h, x:x+w]
        return face_crop

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

user_name = input("Please enter your username: ")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera can't open")
    sys.exit()

base_dir = r'C:\Users\Beyza\goruntu isleme teknikleri\deepface\face_dataset'
user_dir = os.path.join(base_dir, user_name)
create_dir(user_dir)

tour = 0
max_images = 200

while True:
    
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    face_cropped = face_crop(frame)
    if face_cropped is not None:
        tour += 1
        file_path = os.path.join(user_dir, f"{user_name}_{tour}.jpg")
        cv2.imwrite(file_path, face_cropped)
        print(f"Image {tour} saved.")
    cv2.putText(frame, f"Images Captured: {tour}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    fixed_frame = cv2.resize(frame, (640, 480))
    cv2.imshow('frame', fixed_frame)   
    
    

    if cv2.waitKey(1) & 0xFF == ord('q') or tour == max_images:
        break

cap.release()
cv2.destroyAllWindows()
