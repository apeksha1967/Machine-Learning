import cv2

img = cv2.imread('test/test3.jpg')
dataset = cv2.CascadeClassifier('data.xml')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = dataset.detectMultiScale(gray, 1.28)
# we will get x,y,w,h of faces in image

# print(faces)

for x,y,w,h in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 5)

while True:

    cv2.imshow('result',img)
    if cv2.waitKey(10) == 27:
        break

