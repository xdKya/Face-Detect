import cv2

video = cv2.VideoCapture(0)

detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
    sucess, frame = video.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face = detector.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in face:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("webcam", frame)

    if cv2.waitKey(1) == 32:
        break

video.release()
cv2.destroyAllWindows()
