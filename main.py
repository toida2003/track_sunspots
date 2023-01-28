import glob
import cv2
import copy
import numpy as np
import sun_data
import utils
from sklearn.cluster import DBSCAN


def make_sun(color_img, bin_img) -> sun_data.Sun:
    sun = sun_data.Sun(color_img, bin_img)
    sunspots = utils.detect_sunspots(bin_img)
    sun.SetSunspots(sunspots)
    return sun


def main():
    # 画像の読み込み
    img_paths: list[str] = [
        "sun_rotate/2022_12_14.jpg",
        "sun_rotate/2022_12_15.jpg",
        "sun_rotate/2022_12_16.jpg",
    ]
    imgs: list[np.ndarray] = utils.read_imgs(img_paths)

    # 二値化
    imgs_bin: list[np.ndarray] = utils.imgs_binarization(imgs)

    # 背景の黒い部分をトリミングして正方形にする
    (color_imgs, bin_imgs) = utils.clip_imgs(imgs, imgs_bin)

    # Sunクラスにまとめる
    suns: list[sun_data.Sun] = []
    for cimg, bimg in zip(color_imgs, bin_imgs):
        sun = make_sun(cimg, bimg)
        suns.append(sun)

    # クラスタリング
    for i, sun in enumerate(suns):
        sunspots: list[sun_data.Sunspot] = sun.GetSunspotsNorm()
        points: list[tuple] = []
        for sunspot in sunspots:
            points.append(sunspot.GetPoint())

        color = np.random.randint(0, 255, (200, 3))
        min_samples = 4
        eps = 0.2
        dbscan = DBSCAN(min_samples=min_samples, eps=eps)
        clusters = dbscan.fit_predict(points)

        for j, k in enumerate(clusters):
            sun.DrawSunspotImage(j, color[k].tolist())

        cv2.imwrite(
            f"result/{i}_min_samples_{min_samples}_eps_{eps}.jpg",
            sun.GetDrawnSunpotsImage(),
        )


if __name__ == "__main__":
    main()
