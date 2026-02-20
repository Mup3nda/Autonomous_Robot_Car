from .drive import DriveActions
from .vision import VisionActions


class RobotActions:
    def __init__(self, service, gpio, cam, edge):
        self.drive = DriveActions(service, gpio)
        self.vision = VisionActions(cam, edge, gpio, service)
