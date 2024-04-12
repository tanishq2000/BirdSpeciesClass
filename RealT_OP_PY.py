# Here we are putting the whole code inside the function so that we
# can use it as a library and can import it to the program generating the UI
# and call it from there
def class_function():
    # Here we import all the libraries that we may require in this program
    import cv2
    import os
    import pandas as pd
    import csv
    import tensorflow as tf
    import tensorflow.keras.models
    from matplotlib import pyplot as plt
    import numpy as np
    from keras import models
    import sys
    from PIL import Image
    import winsound
    import time
    import math

    # Importing the model that was trained by us using CNN

    model = models.load_model('ModelsRetrain/transferCNN_Retrained(1).h5', compile=False)

    # Importing the CSV file that containes all the threshold values imput by the user

    varFile = pd.read_csv("Var_Values_D.csv")

    # Importing the HaarCascade files to add face landmarks to recognise the location of face
    # in the frame and detect the eye

    faceCascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
    eyeCascade = cv2.CascadeClassifier('haarcascade/haarcascade_eye.xml')

    # Here we are trying to find the required variable values present in the CSV file

    with open('Var_Values_D.csv') as file_obj:
        reader_obj = csv.reader(file_obj)
        for row in reader_obj:
            try:
                if row[1] == 'DrowsiTime':
                    eye_Counter = int(row[2])
                if row[1] == 'SelectEye':
                    index = int(row[2])
            except:
                continue

    # Here we check if the directory in which we intend to save the frames exist or not
    # if not we create that directory

    closed_retrain_output = 'captured_frames/closed'
    open_retrain_output = 'captured_frames/open'
    if not os.path.exists(closed_retrain_output):
        os.makedirs(closed_retrain_output)
    if not os.path.exists(open_retrain_output):
        os.makedirs(open_retrain_output)

    path = "haarcascade/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')

    vid = cv2.VideoCapture(0)
    if not vid.isOpened():
        vid = cv2.VideoCapture(0)
    if not vid.isOpened():
        raise IOError("Can't detect the camera")

    frame_number = 0
    start_time = time.time()
    while vid.isOpened():

        ret, frame = vid.read()
        if not ret:
            break

        # filename = os.path.join(closed_retrain_output, f'frame_{frame_number}.jpg')  # Construct full path
        # cv2.imwrite(filename, frame)
        # frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        text = cv2.FONT_HERSHEY_SIMPLEX
        # Detect the face in the frame
        faces = faceCascade.detectMultiScale(gray, 1.1, 4)

        # eyes = eyeCascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:

            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]


            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            eyes = eyeCascade.detectMultiScale(roi_gray)
            # marked_eye = eyeCascade.detectMultiScale(roi_gray)

            if len(eyes) == 0:
                outPut = "Eye Not Found!"
                print("Eye not found!!")
            elif len(eyes) > 0:
                # Determine which eye to process (left or right)
                # For example, you can choose the first detected eye as the left eye and the second as the right eye
                left_eye = eyes[0]
                right_eye = eyes[1] if len(eyes) > 1 else None  # Check if right eye is detected

                if index == 0:
                    (ex, ey, ew, eh) = left_eye
                    eyes_roi = roi_color[ey: ey + eh, ex:ex + ew]
                    # Draw a rectangle around the detected eye
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
                elif index == 1:
                    (ex, ey, ew, eh) = right_eye if right_eye is not None else left_eye
                    eyes_roi = roi_color[ey: ey + eh, ex:ex + ew]
                    # Draw a rectangle around the detected eye
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
                elif index == 2:
                    for (ex, ey, ew, eh) in eyes:
                        eyes_roi = roi_color[ey: ey + eh, ex:ex + ew]
                        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
                    #for (ex, ey, ew, eh) in right_eye:

                resize_img = cv2.resize(eyes_roi, (224, 224))
                final_img = np.expand_dims(resize_img, axis=0)
                final_pic = final_img / 255.0
                    # final_img = np.expand_dims(final_pic, axis=0)
                prediction = model.predict(final_pic)

                if prediction > 0.5:
                    # filename = f'frame_{frame_number}.jpg'
                    # success = cv2.imwrite(filename, frame)
                    filename = os.path.join(open_retrain_output, f'frame_{frame_number}.jpg')  # Construct full path
                    # filename = "OpenCV_frame{}.png".format(frame_number)
                    capture = cv2.imwrite(filename, eyes_roi)
                    frame_number += 1
                    if capture:
                        print(f"Frame saved successfully as {filename}.")
                    else:
                        print(f"Error: Failed to save frame as {filename}.")

                    outPut = "Open Eye "
                    # cv2.putText(frame, outPut, (20,60), text, 3, (0, 255, 0), 2, cv2.LINE_4)
                    x1, y1, w1, h1 = 0, 0, 180, 75
                    # Drawing a black rectange
                    cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
                    # Adding text
                    cv2.putText(frame, 'Active', (x1 + int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 0), 2)
                    start_time = time.time()
                else:
                    # counter +=1
                    filename = os.path.join(closed_retrain_output, f'frame_{frame_number}.jpg')  # Construct full path
                    capture = cv2.imwrite(filename, eyes_roi)
                    frame_number += 1
                    if capture:
                        print(f"Frame saved successfully as {filename}.")
                    else:
                        print(f"Error: Failed to save frame as {filename}.")

                    outPut = "Closed Eye"
                    # cv2.putText(frame, outPut, (150, 150), text, 3, (0, 0, 255), 2, cv2.LINE_4)
                    x1, y1, w1, h1 = 0, 0, 180, 75
                    cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, 'Active', (x1 + int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 0), 2)
                    if (int(time.time() - start_time) >= eye_Counter):
                        x1, y1, w1, h1 = 0, 0, 180, 75
                        cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
                        cv2.putText(frame, 'Sleep Alert!!', (x1 + int(w1 / 10), y1 + int(h1 / 2)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 255, 0), 2)
                        winsound.Beep(frequency=750, duration=3000)
                        # counter = 0
                # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # print(faceCascade.empty())
                # faces = faceCascade.detectMultiScale(frame, 1.1, 4)

                # for(x, y, w, h) in faces:
                #    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # tut = 'TutorialsPoint'
        image = cv2.putText(frame, outPut, (10, 65), text, 1, (255, 0, 0), 2, cv2.LINE_4)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    cv2.destroyAllWindows()


call = class_function()
