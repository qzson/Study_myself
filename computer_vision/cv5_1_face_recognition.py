# 20-07-21
# 5-1. 얼굴 인식(Face recognition) 정보 추출

# deep learning module을 사용할 것

# Dlib 을 사용하기 위해
# cmake 패키지 설치
# pip install cmake
# pip install dlib
# pip install face_recognition

# 어떤 식으로 face_recognition을 할 것인가 (인식할 데이터 필요)
# 사진은 10~25장씩 권장 준비해서 학습을 시킨다.
# 사진 데이터가 많아질 수록, 선명할 수록 정확도가 증가하고
# 실전에서 정확도를 위해 이미지 선처리 중요 (face_landmark / face_alinement) 을 통해 컴퓨터가 사진들을 인식하기 쉽게 만들어주기도 한다

# 128개의 벡터이미지, real data 인코딩

import cv2
import face_recognition
import pickle

# 상수들을 선언
dataset_paths = ['./AI_CV/source/ai_cv/image/son/', './AI_CV/source/ai_cv/image/tedy/']
names = ['Son', 'Tedy']
number_images = 10  # 10개씩만
image_type = '.jpg'
encoding_file = 'encodings.pickle' # 피클을 사용할 거라서
# Either cnn  or hog. The CNN method is more accurate but slower. HOG is faster but less accurate.
# 중요, cnn 방식 // HOG 방식 사용 (이미지를 검출하고 인식하는데 사용하는 방식 - 아날로그 그림을 디지털화 하는 기술)
# hog 사용한다면 빠른 대신에 정확도가 떨어진다.
model_method = 'cnn'

# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []

# loop over the image paths
for (i, dataset_path) in enumerate(dataset_paths):
    # extract the person name from names
    name = names[i]

    for idx in range(number_images):
        file_name = dataset_path + str(idx+1) + image_type

        # load the input image and convert it from BGR (OpenCV ordering) to dlib ordering (RGB)
        image = cv2.imread(file_name)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # detect the (x, y)-coordinates of the bounding boxes corresponding to each face in the input image
        boxes = face_recognition.face_locations(rgb, model=model_method)

        # compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, boxes)

        # loop over the encodings
        for encoding in encodings:
            # add each encoding + name to our set of known names and encodings
            print(file_name, name, encoding)
            knownEncodings.append(encoding)
            knownNames.append(name)
        
# Save the facial encodings + names to disk
data = {"encodings": knownEncodings, "names": knownNames}
f = open(encoding_file, "wb")
f.write(pickle.dumps(data))
f.close()