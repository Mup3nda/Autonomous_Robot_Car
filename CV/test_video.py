import cv2

cap = cv2.VideoCapture(r'C:\Users\saray\Documents\Master\2_SEM\34755_BDR\project\videos\avi\pi_recording_moving_orange_ball.avi')

print("Opened:", cap.isOpened())
print("FPS:", cap.get(cv2.CAP_PROP_FPS))

while True:
    ret, frame = cap.read()
    if not ret:
        print("No more frames")
        break

    cv2.imshow("Test", frame)
    if cv2.waitKey(200) == 27:   # slow playback to see frames
        break

cap.release()
cv2.destroyAllWindows()