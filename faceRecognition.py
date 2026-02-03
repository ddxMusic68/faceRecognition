import cv2 as cv
import numpy as np
import os

haar = cv.CascadeClassifier('haar_face.xml') # detects face

def fileRenamer(dir_path:str):
    for counter, img in enumerate(os.listdir(dir_path)):
        path = os.path.join(dir_path, img)
        new_file = f"{str(counter+1) + '.' + img.split('.')[1]}"
        new_path = os.path.join(dir_path, new_file)
        os.rename(path,  new_path)

def getFeatures(rel_path: str, face_detect: bool = False):
    """works with jpg's & mp4's"""
    # Gets people
    people = np.array([], dtype="str")
    for dir in os.listdir(os.path.join(os.curdir, rel_path)):
        people = np.append(people, dir)
    np.save('people.npy', people)

    # Featurs and label list
    features = [] # start with python because numpy flattens matrics automatically
    labels = []

    # Loop over every person
    for (counter, person) in enumerate(people):
        person_path = os.path.join(os.curdir, rel_path, person) # os.path.join just builds from .

        # Loop over every Media
        for media in os.listdir(person_path):
            media_path = os.path.join(person_path, media)

            # Image logic
            if media.split('.')[1] == 'jpg':
                img = cv.imread(media_path)
                gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                faces_rect = haar.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=4) # scale factor is how much the scaleing goes up (face could be small or big so this checks both) | minneightbors helps removes false positives

                for (x,y,w,h) in faces_rect:
                    faces_roi = gray_img[y:y+h, x:x+w] # faces_roi stands for faces_region of interest
                    features.append(faces_roi)
                    labels.append(counter)

            # Video logic
            if media.split('.')[1] == 'mp4':
                capture = cv.VideoCapture(media_path)
                while True:
                    isTrue, frame = capture.read()
                    
                    if not isTrue: # breaks out of loop once frame done
                        break

                    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    faces_rect = haar.detectMultiScale(gray_img, 1.1, 4)

                    for (x,y,w,h) in faces_rect:
                        faces_roi = gray_img[y:y+h,x:x+w]
                        features.append(faces_roi)
                        labels.append(counter)
                

    # Save features and labels
    features = np.array(features, dtype='object')
    labels = np.array(labels)

    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    face_recognizer.train(features,labels)
    face_recognizer.save('face_trained.yml') # yml is like a python json


def recongnize_face_img(img:np.ndarray, face_yml:str, people:list[str]):
    # init face_recongnizer
    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    face_recognizer.read(face_yml)

    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # detect the face in the image
    faces_rect = haar.detectMultiScale(gray_img, 1.1, 4)

    for (x,y,w,h) in faces_rect:
        faces_roi = gray_img[y:y+h,x:x+w]

        label, confidence = face_recognizer.predict(faces_roi)

        cv.putText(img, f'{str(people[label])}   {round(confidence, 2)}', (x, y), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
        cv.rectangle(img, (x,y), (x+w,y+h), (0, 255,0), thickness=2)

    cv.imshow('Detected Face', img)
    cv.waitKey(0)
    return label, confidence

def recognize_face_vid(capture:cv.VideoCapture, face_yml:str, people:list[str]):
    # init face_recongnizer
    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    face_recognizer.read(face_yml)
    label = 0
    confidence = 0

    while True:
        isTrue, frame = capture.read()
        gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # detect the face in the image
        faces_rect = haar.detectMultiScale(gray_img, 1.1, 4)

        for (x,y,w,h) in faces_rect:
            faces_roi = gray_img[y:y+h,x:x+w]

            label, confidence = face_recognizer.predict(faces_roi)

            cv.putText(frame, f'{str(people[label])}   {round(confidence, 2)}', (x, y), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
            cv.rectangle(frame, (x,y), (x+w,y+h), (0, 255,0), thickness=2)

        print(people[label], confidence) # DEBUG
        cv.imshow('Detected Face', frame)
        if cv.waitKey(20) & 0xFF==ord('d'): # (stops when d is pressed)
            break


getFeatures('training')
img = cv.imread(r'C:\Users\Rocket\Desktop\Projects\pythonproject2\myprojects\rocketRecognition\validating\jerry_seinfeld/3.jpg')
# cv.imshow("cool", img)
recongnize_face_img(img, 'face_trained.yml', np.load('people.npy', allow_pickle=True))
# recognize_face_vid(cv.VideoCapture(0), 'face_trained.yml', np.load('people.npy', allow_pickle=True))

# path = r'C:\Users\Rocket\Desktop\Projects\pythonproject2\myprojects\rocketRecognition\validating'
# for dir in os.listdir(path):
#     fileRenamer(os.path.join(path, dir))

cv.waitKey(0)