"""Vision/image analysis action commands."""
import cv2 as cv


class VisionActions:
    """Provides image capture and analysis commands.
    
    Args:
        cam: Camera interface for image capture
        edge: Edge sensor interface for line visualization
        gpio: GPIO interface (used for environment checks)
        service: Service interface for configuration and logging
    """
    def __init__(self, cam, edge, gpio, service):
        self.cam = cam
        self.edge = edge
        self.gpio = gpio
        self.service = service

    def image_analysis(self, save):
        """Capture and optionally save an image from the camera.
        
        Draws edge detection visualization on the image, displays on screen
        (if not running on Pi), and optionally saves to disk.
        
        Args:
            save: If True, save the image to disk; if False, just analyze
            
        Returns:
            bool: True if image was successfully captured and processed
        """
        # Check if camera is available
        if not self.cam.useCam:
            return False
            
        # Get current image from camera
        ok, img, img_time = self.cam.getImage()
        if not ok:
            if self.cam.imageFailCnt < 5:
                print("% Failed to get image.")
            return False
            
        # Draw edge detection visualization on image
        self.edge.paint(img)
        
        # Display image on monitor (desktop development only, not on Pi)
        if not self.gpio.onPi:
            try:
                cv.imshow("frame for analysis", img)
            except Exception:
                print("% mqtt-client-mission::image_analysis: failed to show camera image")
        
        # Optionally save image to disk
        if save:
            fn = f"image_{img_time.strftime('%Y_%b_%d_%H%M%S_')}{self.cam.cnt:03d}.jpg"
            cv.imwrite(fn, img)
            if not self.service.args.silent:
                print(f"% Saved image {fn}")
        else:
            print("# imageAnalysis:: image not saved")
        
        return True
