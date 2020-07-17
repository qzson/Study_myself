# 20-07-17
# 3-1. OpenCV DNN 딥러닝 모듈 (dnn 모듈이 이전의 objdetect 모듈보다 성능이 훨씬 좋다.)
# 3-2. dnn 딥러닝 방식을 이용한 Face Detection

import cv2
import numpy as np

model_name = './AI_CV/source/res10_300x300_ssd_iter_140000.caffemodel'     # 모델 (w값 , layer값) 300x300의 이미지
prototxt_name = './AI_CV/source/deploy.prototxt.txt'                       # caffemodel 모델의 설계도
min_confidence = 0.3
file_name= './AI_CV/source/opencv_dnn_202005/image/soccer_02.jpg'

def detectAndDisplay(frame):
    # pass the blob through the model and obtain the detections
    model = cv2.dnn.readNetFromCaffe(prototxt_name, model_name)

    # Resizing to a fixed 300x300 pixels and then normalizing it
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    model.setInput(blob)
    detections = model.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]     # 확률

        # filter out weak detections by ensuring the 'confidence' is greater than the minimum confidence
        if confidence > min_confidence:
            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype('int')
            print(confidence, startX, startY, endX, endY)

            # draw the bounding box of the face along with the associate probability
            text = '{:.2f}%'.format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # show the output image
    cv2.imshow("Face Detection by dnn", frame)

print('OpenCV version:')
print(cv2.__version__)

img = cv2.imread(file_name)
print('w : {} pixels'.format(img.shape[1]))
print('h : {} pixels'.format(img.shape[0]))
print('ch : {}'.format(img.shape[2]))

(height, width) = img.shape[:2]

cv2.imshow('Original Image', img)

detectAndDisplay(img)

cv2.waitKey(0)
cv2.destroyAllWindows()