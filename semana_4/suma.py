import cv2
img1 = cv2.imread('images.jpeg')
img2 = cv2.imread('mik.png')

imagen1 = cv2.resize(img1, (300, 300))
imagen2 = cv2.resize(img2, (300, 300))

# suma = imagen1 + imagen2
suma = cv2.addWeighted(imagen1,0.5, imagen2, 0.5, 3)
# suma = cv2.add(imagen2, imagen1)

#cv2.imshow('Resultado', imagen1)
cv2.imshow('Suma de imagenes', suma)
cv2.waitKey(0)
cv2.destroyAllWindows()

# img1 = cv2.imread('')j
# img2 = cv2.imread('')

# img3 = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)


