import cv2

cap = cv2.VideoCapture(f'http://10.197.218.199:7123/stream.mjpg')
#cap = cv2.VideoCapture(f'http://0.0.0.0:7124/usb_camera')

while True:
    ret, frame = cap.read()
    cv2.imshow("cam", frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

