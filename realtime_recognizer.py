import math
import os

import face_recognition
import cv2
import numpy as np


def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance)/ (range*2.0)

    if face_distance > face_match_threshold :
        return str(np.round(linear_val*100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5)*2, 0.2))) * 100
        return str(np.round(value, 2)) + '%'



# load all known faces
def encode_faces():

    for image in os.listdir('faces'):

        face_image = face_recognition.load_image_file(f'faces/{image}')
        face_encoded = face_recognition.face_encodings(face_image)[0]
        known_face_encodings.append(face_encoded);
        known_names.append(image)

    print( known_names)


known_face_encodings = []
known_names = []
video_capture = cv2.VideoCapture(0)
face_names = []
process_current_frame = True

encode_faces()

while True :
    results, frame =video_capture.read()

    # process every other frame
    if process_current_frame:

        # resize scaling
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_frame = small_frame[:,:,:: -1]


        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        face_names = []

        # loop threw all faces
        for face_encoding in face_encodings:

            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, 0.5)
            name = 'unknown'
            confidence = ''

            face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distance) # choose best match


            if matches[best_match_index]:
                name = known_names[best_match_index]
                confidence = face_confidence(face_distance[best_match_index])

            face_names.append(f'{name} {confidence}')


    process_current_frame = not process_current_frame

    # draw rectangle around all faces
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        color= ()
        top *=4
        right*=4
        bottom*=4
        left*=4

        if name == 'unknown':
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 4)
        cv2.rectangle(frame, (left, bottom- 35), (right, bottom), color, -1)
        cv2.putText(frame, name, (left +6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX,0.8, (255, 255, 255), 1)




    cv2.imshow('Frame', frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

video_capture.release();
cv2.destroyAllWindows()

























