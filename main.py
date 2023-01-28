import glob
import cv2
import copy
import numpy as np
import sun_data
import utils


def make_sun(color_img, bin_img) -> sun_data.Sun:
    sun = sun_data.Sun(color_img, bin_img)
    sunspots = utils.detect_sunspots(bin_img)
    sun.SetSunspots(sunspots)
    return sun


def main():
    # 画像の読み込み
    img_paths = [
        "sun_rotate/2022_12_14.jpg",
        "sun_rotate/2022_12_15.jpg",
        "sun_rotate/2022_12_16.jpg",
    ]
    imgs = utils.read_imgs(img_paths)

    # 二値化
    imgs_bin = utils.imgs_binarization(imgs)

    # 背景の黒い部分をトリミングして正方形にする
    (color_imgs, bin_imgs) = utils.clip_imgs(imgs, imgs_bin)

    # Sunクラスにまとめる
    suns = []
    for cimg, bimg in zip(color_imgs, bin_imgs):
        sun = make_sun(cimg, bimg)
        sun.DrawSunspotsImage()
        suns.append(sun)

    for sun in suns:
        cv2.imshow("img", sun.GetDrawnSunpotsImage())
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
