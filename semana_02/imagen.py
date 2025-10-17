import cv2

img = cv2.imread('miks.png')

if img is None:
    print("Error en la carga de la imagen")

else:
    cv2.imshow('titulo', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    