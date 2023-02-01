import copy
import numpy as np
import cv2


class Sunspot:
    def __init__(self, x, y, w, h) -> None:
        self.x: float = x
        self.y: float = y
        self.w: float = w
        self.h: float = h

        self.upperleft: tuple(int, int) = (int(x - w / 2), int(y - h / 2))
        self.lowerright: tuple(int, int) = (int(x + w / 2), int(y + h / 2))

        self.lat: float = None

    def GetPoint(self) -> tuple:
        return (self.x, self.y)

    def GetArea(self) -> tuple:
        return (self.w, self.h)

    def GetUpperLeft(self) -> tuple:
        return self.upperleft

    def GetLowerRight(self) -> tuple:
        return self.lowerright

    def SetLatitudeRad(self, lat) -> None:
        self.lat = lat

    def GetLatitudeRad(self) -> float:
        return self.lat


class Sun:
    def __init__(
        self,
        color_img: np.ndarray,
        bin_img: np.ndarray,
        size: tuple = (1000, 1000),
    ) -> None:
        self.color_img: np.ndarray = color_img
        self.bin_img: np.ndarray = bin_img
        self.size: tuple(int, int) = size
        self.sunspots: list[Sunspot] = []
        self.sunspots_norm: list[Sunspot] = []
        self.drawn_sunspots_img: np.ndarray = color_img.copy()

    def GetColorImage(self) -> np.ndarray:
        return self.color_img

    def GetBinaryImage(self) -> np.ndarray:
        return self.bin_img

    def SetSunspots(self, sunspots_norm: list) -> None:
        self.sunspots_norm += sunspots_norm
        for sunspot_norm in sunspots_norm:
            p: tuple(float, float) = sunspot_norm.GetPoint()
            s: tuple(float, float) = sunspot_norm.GetArea()
            sunspot: Sunspot = Sunspot(
                p[0] * self.size[0],
                p[1] * self.size[1],
                s[0] * self.size[0],
                s[1] * self.size[1],
            )
            self.sunspots.append(sunspot)

    def GetSunspots(self) -> list[Sunspot]:
        return self.sunspots

    def GetSunspotsNorm(self) -> list[Sunspot]:
        return self.sunspots_norm

    def DrawSunspotImage(self, index: int, color: tuple) -> None:
        self.drawn_sunspots_img = cv2.rectangle(
            self.drawn_sunspots_img,
            self.sunspots[index].GetUpperLeft(),
            self.sunspots[index].GetLowerRight(),
            color,
            2,
        )

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

    def GetImageSize(self) -> tuple:
        return self.size

    def SetSunspotLatitudeRadian(self, index, lad) -> None:
        self.sunspots[index].SetLatitudeRad(lad)
        self.sunspots_norm[index].SetLatitudeRad(lad)

    def GetSunspotLatitudeRadian(self, index) -> float:
        return self.sunspots[index].GetLatitudeRad()


class SunspotsCluster:
    def __init__(self, img: np.ndarray, point: tuple, size=(20, 20)) -> None:
        self.img: np.ndarray = cv2.resize(img, size)
        self.point: tuple(float, float) = point

    def GetImage(self) -> np.ndarray:
        return self.img

    def GetPoint(self) -> tuple:
        return self.point

    def GetPointInteger(self) -> tuple:
        return (int(self.point[0]), int(self.point[1]))
