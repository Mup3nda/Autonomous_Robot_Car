import cv2

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture(f'http://10.197.218.199:7123/stream.mjpg')
cap = cv2.VideoCapture(f'http://0.0.0.0:7124/usb_camera')

while True:
    ret, frame = cap.read()
    cv2.imshow("As capturred", frame)
    h, w, c = frame.shape
    
    
    print(f"Frame: {h, w, c}")
    
    # ch0 = frame[:,:,1]
    # cv2.imshow("Channel One", ch0)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

