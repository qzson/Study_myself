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
# OpenCV에서도 Yolo 구동이 된다. 그러나, CPU만 활용되는 아쉬움이 있다. (그래서 생산단계에서는 사용하기 어려움)