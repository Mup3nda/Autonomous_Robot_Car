import cv2 as cv


class VisionActions:
    def __init__(self, cam, edge, gpio, service):
        self.cam = cam
        self.edge = edge
        self.gpio = gpio
        self.service = service

    def image_analysis(self, save):
        if not self.cam.useCam:
            return False
        ok, img, img_time = self.cam.getImage()
        if not ok:
            if self.cam.imageFailCnt < 5:
                print("% Failed to get image.")
            return False
        self.edge.paint(img)
        if not self.gpio.onPi:
            try:
                cv.imshow("frame for analysis", img)
            except Exception:
                print("% mqtt-client-mission::image_analysis: failed to show camera image")
        if save:
            fn = f"image_{img_time.strftime('%Y_%b_%d_%H%M%S_')}{self.cam.cnt:03d}.jpg"
            cv.imwrite(fn, img)
            if not self.service.args.silent:
                print(f"% Saved image {fn}")
        else:
            print("# imageAnalysis:: image not saved")
        return True
