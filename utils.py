import numpy as np
import cv2
import sun_data


def read_imgs(paths: str, clip_range: int = 100) -> list:
    imgs: list[np.ndarray] = []
    for path in paths:
        img: np.ndarray = cv2.imread(path)

        if clip_range != 0:
            img = img[
                clip_range : img.shape[0] - clip_range,
                clip_range : img.shape[1] - clip_range,
            ]

        imgs.append(img)

    return imgs


def imgs_binarization(imgs: list) -> list:
    imgs_bin: list[np.ndarray] = []
    for img in imgs:
        img_gray: np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_bin: np.ndarray = cv2.adaptiveThreshold(
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

    cliped_color_imgs: list[np.ndarray] = []
    cliped_bin_imgs: list[np.ndarray] = []

    for imgc, imgb in zip(color_imgs, bin_imgs):
        contours, _ = cv2.findContours(
            ~imgb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        x_list: list[int] = []
        y_list: list[int] = []
        for pos in contours[0]:
            x_list.append(pos[0][0])
            y_list.append(pos[0][1])
        upper: int = min(y_list)
        lower: int = max(y_list)
        left: int = min(x_list)
        right: int = max(x_list)

        img_clip: np.ndarray = imgc[upper:lower, left:right]
        img_clip = cv2.resize(img_clip, dsize=size)
        cliped_color_imgs.append(img_clip)

        img_bin_clip: np.ndarray = imgb[upper:lower, left:right]
        img_bin_clip = cv2.resize(img_bin_clip, dsize=size)
        cliped_bin_imgs.append(img_bin_clip)

    return (cliped_color_imgs, cliped_bin_imgs)


def make_annotation(
    contour: list, size: tuple = (1000, 1000)
) -> sun_data.Sunspot:
    x: list(int) = []
    y: list(int) = []
    for pos in contour:
        x.append(pos[0][0])
        y.append(pos[0][1])

    center_x: float = (sum(x) / len(contour)) / size[0]
    center_y: float = (sum(y) / len(contour)) / size[1]
    width: float = (max(x) - min(x)) / size[0]
    height: float = (max(y) - min(y)) / size[1]

    return sun_data.Sunspot(center_x, center_y, width, height)


def detect_sunspots(bin_img: np.ndarray) -> list:
    contours, hierarchy = cv2.findContours(
        bin_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )

    sunspots: list = []
    for contour, h in zip(contours, hierarchy[0]):
        if h[3] != -1:
            sunspots.append(contour)

    sunspots_annotation: list[sun_data.Sunspot] = []
    for sunspot in sunspots:
        sunspots_annotation.append(make_annotation(sunspot))

    return sunspots_annotation
