from deepface import DeepFace
import cv2
import numpy as np 
import os
import matplotlib.pyplot as plt



backends=["opencv","ssd","dlib","mtcnn","retinaface"]
models=["VGG-Face","Facenet","Facenet512","OpenFace","DeepFace","DeepID","ArcFace","Dlib","SFace"]
metrics=["cosine","euclidean","euclidean_12"]

img = DeepFace.extract_faces("face_dataset//beyza//beyza_55.jpg")



def face_recog(img):
    results=DeepFace.find(img_path=img,db_path="face_dataset",model_name=models[4],distance_metric=metrics[1],enforce_detection=False)
    plt.imshow(cv2.imread(img))
     
    
    for person in results:
        print(person['identitiy'][0].split('/')[1])

def real_time_recog():
    vid=cv2.VideoCapture(0)

    while True:
        ret,frame=vid.read()

        results=DeepFace.find(img_path=frame,db_path="face_dataset",model_name=models[2],distance_metric=metrics[1],enforce_detection=False)
        for result in results:
            x=result['x'][0]
            y=result['y'][0]
            w=result['w'][0]
            h=result['h'][0]

            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            name=result['identity'][0].split('/')[1]
            cv2.putText(frame,name,(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
        cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('frame',960,720)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1)& 0xFF==ord('q'):
            break
    vid.release()
    cv2.destroyAllWindows()


real_time_recog()






