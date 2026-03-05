import cv2
import numpy as np

Kp_turn = 0.004
Kp_fwd  = 0.01

r_target = 80  

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W = frame.shape[:2]
    center_x = W // 2

    frame = cv2.GaussianBlur(frame,(11,11),0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #red color range in HSV
    lower1 = (0,120,70)
    upper1 = (10,255,255)

    lower2 = (170,120,70)
    upper2 = (180,255,255)
    
    #dark red color range in HSV
    lower1 = (0,120,40)
    upper1 = (10,255,255)

    lower2 = (170,120,40)
    upper2 = (180,255,255)
    #white color range in HSV
    lower_white1 = (0, 0, 200)
    upper_white1 = (10, 50, 255)
    lower_white2 = (170, 0, 200)
    upper_white2 = (180, 50, 255)

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower1, upper1)
    mask = mask1 | mask2
    
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    cv2.imshow("mask", mask)

    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    forward = 0
    angular = 0

    if len(contours) > 0:

        c = max(contours, key=cv2.contourArea)

        area = cv2.contourArea(c)
        if area > 100:

            (xr,yr,wr,hr) = cv2.boundingRect(c)
            rect_area = wr * hr
            fill_ratio = area / rect_area

            if fill_ratio > 0.5:

                perimeter = cv2.arcLength(c, True)

                if perimeter > 0:

                    circularity = 4*np.pi*area/(perimeter*perimeter)

                    if circularity > 0.8:

                        (cx,cy),radius = cv2.minEnclosingCircle(c)

                        err_x = cx - center_x

                        angular = -Kp_turn * err_x
                        forward = Kp_fwd * (r_target - radius)

                        angular = max(min(angular, 1.0), -1.0)
                        forward = max(min(forward, 0.5), 0)

                        cv2.circle(frame,(int(cx),int(cy)),int(radius),(0,255,0),2)
                        cv2.putText(frame,"RED BALL",
                                    (int(cx)-40,int(cy)-20),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,(0,255,0),2)

                        cv2.line(frame,(center_x,0),(center_x,H),(255,0,0),2)

                        cv2.arrowedLine(frame,(50,H-50),
                                        (50,int(H-50-100*forward)),
                                        (0,255,0),3)

                        turn_x = int(W/2 + angular*200)
                        cv2.line(frame,(W//2,H-30),(turn_x,H-30),(0,0,255),4)

    text = f"forward={forward:.2f}  angular={angular:.2f}"
    cv2.putText(frame,text,(20,30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,(255,255,255),2)

    cv2.imshow("Robot Simulation", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
