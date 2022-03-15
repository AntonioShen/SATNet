import cv2
import numpy as np


def visualize(data, is_input, label):
    img = np.zeros((1000, 3000))
    data = data.reshape((81, 18))
    is_input = is_input.reshape((81, 18))
    label = label.reshape((81, 9))
    x, y = np.where(data == 1)
    _data = np.zeros(81)
    _data[x] = y + 1
    data = _data
    for i in range(9):
        for j in range(9):
            cv2.putText(img, str(int(data[i * 9 + j])), (50 + j * 100, 50 + i * 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    x = np.any(is_input == 1, axis=1)
    _is_input = np.zeros(81)
    _is_input[x] = 1
    is_input = _is_input
    for i in range(9):
        for j in range(9):
            cv2.putText(img, str(int(is_input[i * 9 + j])), (1050 + j * 100, 50 + i * 100), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (255, 255, 255), 2)
    x, y = np.where(label == 1)
    _label = np.zeros(81)
    _label[x] = y + 1
    label = _label
    for i in range(9):
        for j in range(9):
            cv2.putText(img, str(int(label[i * 9 + j])), (2050 + j * 100, 50 + i * 100), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (255, 255, 255), 2)
    cv2.imwrite('draw.png', img)
    exit(0)
