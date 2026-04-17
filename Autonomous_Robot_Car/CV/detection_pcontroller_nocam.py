import cv2
import numpy as np

Kp_turn = 0.004
Kp_fwd  = 0.01
r_target = 80
num_balls = 6
locked_target = None
LOCK_DISTANCE = 120

cap = cv2.VideoCapture(r'C:\Users\saray\Documents\Master\2_SEM\34755_BDR\project\videos\pi_recording_moving_orange_ball.mp4')

if not cap.isOpened():
    print("ERROR: video not opened")

def ball_priority(ball):
    _, cx, cy, radius, color = ball
    distance_score = radius
    color_weight = {
        "orange_golf": 4,
        "red": 3,
        "blue": 2,
        "white": 1
    }.get(color, 0)
    return distance_score * 10 + color_weight


while True:

    ret, frame = cap.read()
    if not ret:
        print("End of video")
        break

    H, W = frame.shape[:2]
    center_x = W // 2

    # ---------- PREPROCESS ----------
    frame = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # =====================================================
    # ================= COLOR MASKS =======================
    # =====================================================

    # ORANGE (golf ball) - más estrecho para no confundir con rojo
    lower_orange = np.array([10, 120, 80])
    upper_orange = np.array([28, 255, 255])
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)

    # RED (3D printed)
    lower_red1 = np.array([0, 120, 80])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 80])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

    # BLUE (3D printed)
    lower_blue = np.array([90, 60, 60])
    upper_blue = np.array([135, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # WHITE (3D printed) - tolerante a sombras
    lower_white = np.array([0, 0, 90])
    upper_white = np.array([180, 60, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # =====================================================
    # ================ MORPHOLOGY (PER MASK) ==============
    # =====================================================

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    mask_orange = cv2.morphologyEx(mask_orange, cv2.MORPH_OPEN, kernel)
    mask_orange = cv2.morphologyEx(mask_orange, cv2.MORPH_CLOSE, kernel2)

    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel2)

    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel2)

    # blanco: closing más fuerte para unir zonas en sombra
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel_open)
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel_close)

    # DEBUG: solo para ver
    debug_mask = mask_orange | mask_red | mask_blue | mask_white
    cv2.imshow("mask_debug", debug_mask)

    forward = 0
    angular = 0
    valid_balls = []

    # =====================================================
    # =============== BALL DETECTION BY COLOR =============
    # =====================================================

    color_masks = [
        ("orange_golf", mask_orange),
        ("red", mask_red),
        ("blue", mask_blue),
        ("white", mask_white)
    ]

    for color_name, mask in color_masks:

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:

            area = cv2.contourArea(c)
            if area < 300 or area > 12000:
                continue

            x, y, w, h = cv2.boundingRect(c)
            rect_area = w * h
            if rect_area == 0:
                continue

            fill_ratio = area / rect_area
            if fill_ratio < 0.35:
                continue

            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)

            aspect = w / float(h)

            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue
            solidity = area / hull_area

            # ---------- FILTERS PER COLOR ----------
            if color_name == "orange_golf":
                # golf ball: más circular
                if circularity < 0.78:
                    continue
                if aspect < 0.80 or aspect > 1.25:
                    continue
                if solidity < 0.90:
                    continue
            else:
                # 3D printed balls: pueden ser “elípticas”
                if circularity < 0.7:
                    continue
                if aspect < 0.60 or aspect > 1.60:
                    continue
                if solidity < 0.85:
                    continue

            (cx, cy), radius = cv2.minEnclosingCircle(c)

            cx_i, cy_i = int(cx), int(cy)
            if cy_i >= H or cx_i >= W:
                continue

            valid_balls.append((c, cx, cy, radius, color_name))

    valid_balls = sorted(valid_balls, key=lambda x: x[3], reverse=True)[:num_balls]

    # ---------- DRAW DETECTIONS ----------
    for (_, cx, cy, radius, color_name) in valid_balls:
        cv2.circle(frame, (int(cx), int(cy)), int(radius), (0, 255, 0), 2)
        cv2.putText(frame, color_name,
                    (int(cx) - 40, int(cy) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

    # =====================================================
    # =============== TARGET LOCK & CONTROL ===============
    # =====================================================

    if len(valid_balls) > 0:

        if locked_target is None:
            locked_target = max(valid_balls, key=ball_priority)
        else:
            _, prev_cx, prev_cy, _, _ = locked_target
            best_match = None
            best_dist = 99999

            for b in valid_balls:
                _, cx, cy, _, _ = b
                dist = np.hypot(cx - prev_cx, cy - prev_cy)

                if dist < best_dist and dist < LOCK_DISTANCE:
                    best_dist = dist
                    best_match = b

            locked_target = best_match

    if locked_target is not None:

        _, cx, cy, radius, target_color = locked_target

        err_x = cx - center_x

        angular = -Kp_turn * err_x
        forward = Kp_fwd * (r_target - radius)

        angular = max(min(angular, 1.0), -1.0)
        forward = max(min(forward, 0.5), 0)

        cv2.line(frame, (center_x, 0), (center_x, H), (255, 0, 0), 2)
        cv2.arrowedLine(frame, (50, H - 50),
                        (50, int(H - 50 - 100 * forward)),
                        (0, 255, 0), 3)

        turn_x = int(W / 2 + angular * 200)
        cv2.line(frame, (W // 2, H - 30), (turn_x, H - 30), (0, 0, 255), 4)

        cv2.putText(frame, f"LOCKED: {target_color}",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2)

    text = f"forward={forward:.2f}  angular={angular:.2f}"
    cv2.putText(frame, text, (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)

    cv2.imshow("Robot Simulation", frame)

    if cv2.waitKey(200) == 27:
        break

cap.release()
cv2.destroyAllWindows()