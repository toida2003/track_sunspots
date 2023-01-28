import copy
import numpy as np
import cv2


class Sunspot:
    def __init__(self, x, y, w, h) -> None:
        self.x = x
        self.y = y
        self.w = w
        self.h = h

        self.upperleft = (int(x - w / 2), int(y - h / 2))
        self.lowerright = (int(x + w / 2), int(y + h / 2))

    def GetPoint(self) -> tuple:
        return (self.x, self.y)

    def GetArea(self) -> tuple:
        return (self.w, self.h)

    def GetUpperLeft(self) -> tuple:
        return self.upperleft

    def GetLowerRight(self) -> tuple:
        return self.lowerright


class Sun:
    def __init__(
        self,
        color_img: np.ndarray,
        bin_img: np.ndarray,
        size: tuple = (1000, 1000),
    ) -> None:
        self.color_img = color_img
        self.bin_img = bin_img
        self.size = size
        self.sunspots = []
        self.sunspots_norm = []
        self.drawn_sunspots_img = color_img.copy()

    def GetColorImage(self) -> np.ndarray:
        return self.color_img

    def GetBinaryImage(self) -> np.ndarray:
        return self.bin_img

    def SetSunspots(self, sunspots_norm: list) -> None:
        self.sunspots_norm += sunspots_norm
        for sunspot_norm in sunspots_norm:
            p = sunspot_norm.GetPoint()
            s = sunspot_norm.GetArea()
            sunspot = Sunspot(
                p[0] * self.size[0],
                p[1] * self.size[1],
                s[0] * self.size[0],
                s[1] * self.size[1],
            )
            self.sunspots.append(sunspot)

    def GetSunspots(self) -> list:
        return self.sunspots

    def GetSunspotsNorm(self) -> list:
        return self.sunspots_norm

    def DrawSunspotsImage(self, color: tuple = (255, 0, 0)) -> None:
        for sunspot in self.sunspots:
            self.drawn_sunspots_img = cv2.rectangle(
                self.drawn_sunspots_img,
                sunspot.GetUpperLeft(),
                sunspot.GetLowerRight(),
                color,
                2,
            )

    def GetDrawnSunpotsImage(self) -> np.ndarray:
        return self.drawn_sunspots_img
