# 20-07-17
# YOLO란 무엇인가?

'''
YOLO : Real-Time Object Detection
you only look once (YOLO)

YOLO 가 등장할 당시에 오브젝트 디텍션은 주로 Faster R-CNN (Region with Convolutional Neural Nerwork) 계열이 좋은 성능을 내고 있었다.
이때 YOLO 가 등장하여 45프레임을 보여주었고 빠른 버전의 경우 155프레임을 기록하며 사람들을 놀라게 했다.
R-CNN - Fast R-CNN - Faster R-CNN - YOLO 는 대략 10배씩 속도차이가 난다.
게다가 성능도 Faster R-CNN에 비해 크게 떨어지지 않았다.

'''

# 4-2. YOLO 사물 식별(Object Detection) 프로그램

# Yolo는 사실 파이썬에서 구동하기엔 많은 불편함이 따른다.
# Yolo는 소프트웨어가 아니기 때문에, 그것을 제대로 구현하기 위한 프레임워크가 필요하다 (Darknet)
# 그러나, Darknet은 리눅스 기반이며, TensorFlow와 결합(darkflow)시, 상당히 복잡해진다.
# 해당 강의에서는 Darkflow는 사용하지 않을 것이다.
# OpenCV 에서도 Yolo 구동이 된다. 그러나, CPU만 활용되는 아쉬움이 있다. (그래서 생산단계에서는 사용하기 어려움)


import cv2
import numpy as np

min_confidence = 0.5

# Load Yolo
net = cv2.dnn.readNet('./AI_CV/source/opencv_dnn_202005/yolo/yolov3.weights', './AI_CV/source/opencv_dnn_202005/yolo/yolov3.cfg')
classes = []
with open("./AI_CV/source/opencv_dnn_202005/yolo/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
img = cv2.imread("./AI_CV/source/opencv_dnn_202005/image/yolo_01.jpg")
# img = cv2.resize(img, None, fx=0.4, fy=0.4)
height, width, channels = img.shape
cv2.imshow("Original Image", img)

# Detecting objects (blob 형태로 만들 때, yolo는 416, 416 는 acc와 퍼포먼스의 중간단계에 있는 크기)
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

net.setInput(blob)
outs = net.forward(output_layers)

# Showing informations on the screen (배열에 넣는다 - )
class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > min_confidence:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        print(i, label)
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y + 30), font, 2, (0, 255, 0), 1)

cv2.imshow("YOLO Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
