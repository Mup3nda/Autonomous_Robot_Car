#!/usr/bin/env python3
"""
Example usage of the TargetDetector interface with SBall implementation.
"""

from sball_saray import SBall
from target_detector import TargetDetector

def main():
    # Create a ball detector (which implements TargetDetector)
    detector = SBall(cam=None, gpio=None, service=None)

    # Set detection color
    detector.set_detection_color("blue")

    # Start the detector
    detector.start()

    print("Target Detector Example")
    print("======================")

    # Simulate some detection (in real usage, this would run continuously)
    # Here we're just showing the interface methods

    # Check if target is visible
    if detector.is_target_visible(min_confidence=1):
        print("✓ Target detected!")

        # Get target information
        target = detector.get_target()
        if target:
            print(f"  Position: ({target['x']}, {target['y']})")
            print(f"  Radius: {target['radius']} pixels")
            print(f"  Color: {target['color']}")
            print(f"  Confidence: {target['confidence']}/20")
    else:
        print("✗ No target detected")

    # Get full status
    status = detector.get_status()
    print(f"\nSystem Status:")
    print(f"  Running: {status['system_running']}")
    print(f"  Detection Color: {status['detection_color']}")
    print(f"  Image Size: {status['image_size']}")

    # Stop the detector
    detector.stop()
    print("\nTarget detector stopped.")

if __name__ == "__main__":
    main()