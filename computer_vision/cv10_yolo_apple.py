# 원하는 이미지로 YOLO 데이터 만들기
# 학습 데이터 만들기
# 

import os

current_path = os.path.abspath(os.curdir)
DARKNET_ESCAPE_PATH = 'D:\Study_myself'
DARKNET_PATH = 'D:\Study_myself'

# DATA_PATH = 'images'
YOLO_IMAGE_PATH = current_path + '/computer_vision/source/apple/openCV_dnn_v2/custom_self'
YOLO_FORMAT_PATH = current_path + '/computer_vision/source/apple/openCV_dnn_v2/custom_self'
print('\n', YOLO_IMAGE_PATH) # D:\Study_myself/computer_vision/source/apple/openCV_dnn_v2/custom_self

class_count = 0
test_percentage = 0.2
paths = []

with open(YOLO_FORMAT_PATH + '/' + 'classes.names', 'w') as names, \
     open(YOLO_FORMAT_PATH + '/' + 'classes.txt', 'r') as txt:
    for line in txt:
        names.write(line)  
        class_count += 1
    print ("[classes.names] is created")

with open(YOLO_FORMAT_PATH + '/' + 'custom_data.data', 'w') as data:
    data.write('classes = ' + str(class_count) + '\n')
    data.write('train = ' + DARKNET_ESCAPE_PATH + '/computer_vision/source/apple/openCV_dnn_v2/custom_self/' + 'train.txt' + '\n')
    data.write('valid = ' + DARKNET_ESCAPE_PATH + '/computer_vision/source/apple/openCV_dnn_v2/custom_self/' + 'test.txt' + '\n')
    data.write('names = ' + DARKNET_ESCAPE_PATH + '/computer_vision/source/apple/openCV_dnn_v2/custom_self/' + 'classes.names' + '\n')
    data.write('backup = backup')
    print ("[custom_data.data] is created")

os.chdir(YOLO_IMAGE_PATH)
for current_dir, dirs, files in os.walk('.'):
    for f in files:
        if f.endswith('.jpg'):
            image_path = DARKNET_PATH + '/computer_vision/source/apple/openCV_dnn_v2/custom_self/' + f
            paths.append(image_path + '\n')


paths_test = paths[:int(len(paths) * test_percentage)]

paths = paths[int(len(paths) * test_percentage):]


with open(YOLO_FORMAT_PATH + '/' + 'train.txt', 'w') as train_txt:
    for path in paths:
        train_txt.write(path)
    print ("[train.txt] is created")

with open(YOLO_FORMAT_PATH + '/' + 'test.txt', 'w') as test_txt:
    for path in paths_test:
        test_txt.write(path)
    print ("[test.txt] is created")

