import cv2
from cv2 import aruco

# Crear marcador
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
marker = aruco.generateImageMarker(aruco_dict, 0, 200)
cv2.imwrite("marcador.png", marker)