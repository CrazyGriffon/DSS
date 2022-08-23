import math
from operator import itemgetter

import cv2
#import pafy
import numpy as np
from pytube import YouTube


def load_model():
    # load our serialized model from disk
    print("[INFO] loading model...")
    proto_path = 'model/MobileNetSSD_deploy.prototxt'
    model_path = 'model/MobileNetSSD_deploy.caffemodel'
    cvNet = cv2.dnn.readNetFromCaffe(proto_path, model_path)

    # initialize the list of class labels MobileNet SSD was trained to detect
    classes = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    return cvNet, classes


"""
def load_video_from_youtube(url):
	video = pafy.new(url, ydl_opts={'nocheckcertificate': True}, size=True, basic=True)
	return video.getbestvideo()
"""


def get_wingspan_pix(gs):
    _, th2 = cv2.threshold(gs, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(th2, 8, cv2.CV_32S)
    top = stats[0, cv2.CC_STAT_LEFT]
    left = stats[0, cv2.CC_STAT_TOP]
    w = stats[0, cv2.CC_STAT_WIDTH]
    h = stats[0, cv2.CC_STAT_HEIGHT]
    #	cv2.imshow('th2', th2)
    (cX, cY) = centroids[0]
    return top, left, w, h, cX, cY


def add(x, y):
    return [x[i] + y[i] for i in range(len(x))]


# загрузка модели
cvNet, CLASSES = load_model()

# получаем видео c наилучшего возможного качества с YouTube c помощью библиотеки pafy
url = "https://www.youtube.com/watch?v=APiuutE4ac0"

# цвет надписей на изображении
COLOR = [0, 0, 0]
# порог отбрасывания ложных срабатываний модели.Взят низким, чтобы для не очень хорошего качества видео максимально детектировать объект
confidence_value = 0.01

# Изображение самолета мало относительно размера кадра. Если фрейм 720x1280 уменьшать до 300x300(размер входных данных модели),
# то объект исказится по размерам и качество детектирования будет хуже. Поэтому сначала детектируем во всем кадре, а далее ищем в
# окрестности радиуса radius c центром center найденного в прошлом кадре объекта
radius = 150  # pixels
center = None

# ядро фильтра повышения резкости
kernel = np.array([[-1, -1, -1],
                   [-1, 9, -1],
                   [-1, -1, -1]])

# размах крыльев самолета, м (в некоторых источниках - 11.39)
wingspan_real = 11.36

# фокусное расстояние, м
F = 1.8e-3

# диагональ матрицы, м
sensor_diag = 5.64e-3

# радиус окрестности, пикс
margin_pix = 10

# находим размер матрицы
angle = math.atan(9.0 / 16)
sensor_x = sensor_diag * math.cos(angle)
sensor_y = sensor_diag * math.sin(angle)

idx_of_plane = CLASSES.index("aeroplane")

yt = YouTube(url)
best = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()

# main loop
capture = cv2.VideoCapture(best.url)
first_pass = True

while capture.isOpened():
    # получили фрейм
    is_ok, frame = capture.read()
    if (is_ok == False):
        print('Bad frame. Ending reading loop...')
        break

    if first_pass:
		#на первой итерации получаем размеры кадра
        frame_y_pix, frame_x_pix = frame.shape[:2]
        screen_center_x, screen_center_y = frame_x_pix / 2.0, frame_y_pix / 2.0
        first_pass = False

    subframe = frame
    # будем искать объект сначала во всем фрейме, а потом в окрестности объекта, найденного в прошлый раз
    # center-центр окрестности радиуса radius, в которой ищем объект
    if (center != None):
        subframe = frame[center[1] - radius:center[1] + radius, center[0] - radius:center[0] + radius]

    # фильтрация для увеличения четкости
    subframe = cv2.filter2D(subframe, -1, kernel)
    # перевод в grayscale
    grayscale = cv2.cvtColor(subframe, cv2.COLOR_RGB2GRAY)
    # опытным путем установлено, что поиск работает лучше для изображения, все 3 канала которого = grayscale
    subframe = np.dstack([grayscale, grayscale, grayscale]).astype(np.uint8)

    # grab the frame dimensions and convert it to a blob
    (h, w) = subframe.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(subframe, (300, 300)), 0.007843, (300, 300), 127.5)

    # отправляем изображение на вход модели
    cvNet.setInput(blob)
    # находим объекты
    detections = cvNet.forward()

    # выберем только самолеты
    plane_detections = [x for x in detections[0][0] if x[1] == idx_of_plane]

    if (len(plane_detections) >= 1):
        # выберем объект с максимальным скорингом
        plane_detection = max(plane_detections, key=itemgetter(2))
        confidence = plane_detection[2]

        # выберем надежные детектирования
        if confidence > confidence_value:
            box = plane_detection[3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # draw the prediction on the frame
            label = "{}: {:.2f}%".format(CLASSES[idx_of_plane],
                                         confidence * 100)

            # добавим отступы в найденный bounding box
            startX, endX, startY, endY = add([startX, endX, startY, endY],
                                             [-margin_pix, margin_pix, -margin_pix, margin_pix])

            # уточняем положение самолета для увеличеня точности расчета
            top, left, plane_w_pix, plane_h_pix, plane_center_x_, plane_center_y_ = get_wingspan_pix(
                grayscale[startY:endY, startX:endX])

            # пересчет координат относительно исходного фрейма
            startX += top
            endX = startX + plane_w_pix
            startY += left
            endY = startY + plane_h_pix
            plane_center_x = (startX + endX) / 2.0
            plane_center_y = (startY + endY) / 2.0

            if center != None:
                startX += (center[0] - radius)
                endX = startX + plane_w_pix
                startY += (center[1] - radius)
                endY = startY + plane_h_pix
                plane_center_x += (center[0] - radius)
                plane_center_y += (center[1] - radius)
            # обновляем центр самолета
            center = [math.floor(plane_center_x), math.floor(plane_center_y)]

            # определение расстояния и отклонения
            wingspan_sensor = sensor_x / frame_x_pix * plane_w_pix
            d = F * (1 + wingspan_real / wingspan_sensor)
            x_deviation = (screen_center_x - plane_center_x) * wingspan_real / plane_w_pix
            y_deviation = (screen_center_y - plane_center_y) * wingspan_real / plane_w_pix

            # отрисовка
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLOR, 1)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 1)

            cv2.putText(frame, f'distance: {d}', (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 1)
            cv2.putText(frame, f'X deviation: {x_deviation}', (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 1)
            cv2.putText(frame, f'Y deviation: {y_deviation}', (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 1)
    else:
        center = None
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
