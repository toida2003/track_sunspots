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
    clusters_list = []
    for i, sun in enumerate(suns):
        sunspots: list[sun_data.Sunspot] = sun.GetSunspotsNorm()
        points: list[tuple] = []
        for sunspot in sunspots:
            points.append(sunspot.GetPoint())

        color = np.random.randint(0, 255, (200, 3))
        min_samples = 3
        eps = 0.2
        dbscan = DBSCAN(min_samples=min_samples, eps=eps)
        clusters = dbscan.fit_predict(points)
        clusters_list.append(clusters)

        for j, k in enumerate(clusters):
            sun.DrawSunspotImage(j, color[k].tolist())

        cv2.imwrite(
            f"result/{i}_min_samples_{min_samples}_eps_{eps}.jpg",
            sun.GetDrawnSunpotsImage(),
        )

    clusters_sunspots: list[list[sun_data.SunspotsCluster]] = []
    for sun, clusters in zip(suns, clusters_list):
        sunspots: list[sun_data.Sunspot] = sun.GetSunspots()
        cluster_types = []
        for cluster in clusters:
            is_resisterd = False
            for i in cluster_types:
                if i == cluster:
                    is_resisterd = True
            if not (is_resisterd):
                cluster_types.append(cluster)

        cluster_sunspots = []
        for cluster_type in cluster_types:
            sunspots_cluster: list[sun_data.Sunspot] = []
            for sunspot, cluster in zip(sunspots, clusters):
                if cluster == cluster_type:
                    sunspots_cluster.append(sunspot)

            x_list = []
            y_list = []
            for sunspot in sunspots_cluster:
                point = sunspot.GetPoint()
                area = sunspot.GetArea()
                x_list.append(point[0] + area[0] / 2)
                x_list.append(point[0] - area[0] / 2)
                y_list.append(point[1] + area[1] / 2)
                y_list.append(point[1] - area[1] / 2)
            upper = int(min(y_list))
            lower = int(max(y_list))
            left = int(min(x_list))
            right = int(max(x_list))

            clip_cluster = sun.GetColorImage().copy()
            clip_cluster = clip_cluster[upper:lower, left:right]
            cluster_x = (right - left) / 2
            cluster_y = (lower - upper) / 2
            cluster_sunspots.append(
                sun_data.SunspotsCluster(clip_cluster, (cluster_x, cluster_y))
            )

        clusters_sunspots.append(cluster_sunspots)


if __name__ == "__main__":
    main()
