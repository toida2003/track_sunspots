import glob
import cv2
import copy
import numpy as np


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
        self.drawn_sunspots_img = color_img

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
            print(sunspot.GetUpperLeft())
            print(sunspot.GetLowerRight())
            self.drawn_sunspots_img = cv2.rectangle(
                self.drawn_sunspots_img,
                sunspot.GetUpperLeft(),
                sunspot.GetLowerRight(),
                color,
                2,
            )

    def GetDrawnSunpotsImage(self) -> np.ndarray:
        return self.drawn_sunspots_img


def read_imgs(paths: str, clip_range: int = 100) -> list:
    imgs = []
    for path in paths:
        img = cv2.imread(path)

        if clip_range != 0:
            img = img[
                clip_range : img.shape[0] - clip_range,
                clip_range : img.shape[1] - clip_range,
            ]

        imgs.append(img)

    return imgs


def imgs_binarization(imgs: list) -> list:
    imgs_bin = []
    for img in imgs:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_bin = cv2.adaptiveThreshold(
            img_gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            51,
            20,
        )

        img_bin = cv2.medianBlur(img_bin, 5)

        imgs_bin.append(img_bin)

    return imgs_bin


def clip_imgs(
    color_imgs: list, bin_imgs: list, size: tuple = (1000, 1000)
) -> tuple:

    cliped_color_imgs = []
    cliped_bin_imgs = []

    for imgc, imgb in zip(color_imgs, bin_imgs):
        contours, _ = cv2.findContours(
            ~imgb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        x_list = []
        y_list = []
        for pos in contours[0]:
            x_list.append(pos[0][0])
            y_list.append(pos[0][1])
        upper = min(y_list)
        lower = max(y_list)
        left = min(x_list)
        right = max(x_list)

        img_clip = imgc[upper:lower, left:right]
        img_clip = cv2.resize(img_clip, dsize=size)
        cliped_color_imgs.append(img_clip)

        img_bin_clip = imgb[upper:lower, left:right]
        img_bin_clip = cv2.resize(img_bin_clip, dsize=size)
        cliped_bin_imgs.append(img_bin_clip)

    return (cliped_color_imgs, cliped_bin_imgs)


def make_annotation(contour: list, size: tuple = (1000, 1000)) -> Sunspot:
    x = []
    y = []
    for pos in contour:
        x.append(pos[0][0])
        y.append(pos[0][1])

    center_x = (sum(x) / len(contour)) / size[0]
    center_y = (sum(y) / len(contour)) / size[1]
    width = (max(x) - min(x)) / size[0]
    height = (max(y) - min(y)) / size[1]

    return Sunspot(center_x, center_y, width, height)


def detect_sunspots(bin_img: np.ndarray) -> list:
    contours, hierarchy = cv2.findContours(
        bin_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )

    sunspots = []
    for contour, h in zip(contours, hierarchy[0]):
        if h[3] != -1:
            sunspots.append(contour)

    sunspots_annotation = []
    for sunspot in sunspots:
        sunspots_annotation.append(make_annotation(sunspot))

    return sunspots_annotation


def make_sun(color_img, bin_img) -> Sun:
    sun = Sun(color_img, bin_img)
    sunspots = detect_sunspots(bin_img)
    sun.SetSunspots(sunspots)
    return sun


def main():
    img_paths = [
        "sun_rotate/2022_12_14.jpg",
        "sun_rotate/2022_12_15.jpg",
        "sun_rotate/2022_12_16.jpg",
    ]
    imgs = read_imgs(img_paths)
    imgs_bin = imgs_binarization(imgs)

    (color_imgs, bin_imgs) = clip_imgs(imgs, imgs_bin)

    suns = []
    for cimg, bimg in zip(color_imgs, bin_imgs):
        sun = make_sun(cimg, bimg)
        suns.append(sun)

    for sun in suns:
        sun.DrawSunspotsImage()
        cv2.imshow("img", sun.GetDrawnSunpotsImage())
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
