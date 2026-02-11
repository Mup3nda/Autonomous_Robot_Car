from picamera2 import Picamera2, Preview 
import time
import cv2

picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(main={"size":(640, 380), "format":"RGB888"})
picam2.configure(camera_config)

#picam2.start_preview(Preview.QTGL)
picam2.start()

while True:
    frame = picam2.capture_array()
    cv2.imshow("Frame", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#print("Capture Image")
#picam2.capture_file("test.jpg")
picam2.stop()
cv2.destroyAllWindows()
