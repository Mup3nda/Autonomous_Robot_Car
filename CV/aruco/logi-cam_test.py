import cv2

cap = cv2.VideoCapture(f'http://10.197.218.199:7123/stream.mjpg')

while True:
    ret, frame = cap.read()
    cv2.imshow("cam", frame)
    if cv2.waitKey(1) == 27:
        break
cv2.release()
cv2.destroyAllWindows()