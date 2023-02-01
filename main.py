import cv2
import numpy as np
import math
import sun_data
import utils
from sklearn.cluster import DBSCAN


def calc_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    error: float = np.sum((img1.astype(float) - img2.astype(float)) ** 2)
    mse: float = error / (float(img1.shape[0] * img1.shape[1]))
    return mse


def make_sun(color_img: np.ndarray, bin_img: np.ndarray) -> sun_data.Sun:
    sun: sun_data.Sun = sun_data.Sun(color_img, bin_img)
    sunspots: list = utils.detect_sunspots(bin_img)
    sun.SetSunspots(sunspots)
    return sun


def main():
    # 画像の読み込み
    img_paths: list[str] = [
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
    clusters_list: list[list] = []
    for i, sun in enumerate(suns):
        sunspots: list[sun_data.Sunspot] = sun.GetSunspotsNorm()
        points: list[tuple] = []
        for sunspot in sunspots:
            points.append(sunspot.GetPoint())

        color: tuple = np.random.randint(0, 255, (200, 3))
        min_samples: int = 3
        eps: float = 0.2
        dbscan = DBSCAN(min_samples=min_samples, eps=eps)
        clusters: list = dbscan.fit_predict(points)
        clusters_list.append(clusters)

        for j, k in enumerate(clusters):
            sun.DrawSunspotImage(j, color[k].tolist())

    # クラスタリングした範囲を切り抜き
    clusters_sunspots: list[list[sun_data.SunspotsCluster]] = []
    for sun, clusters in zip(suns, clusters_list):
        sunspots: list[sun_data.Sunspot] = sun.GetSunspots()
        cluster_types: list = []
        for cluster in clusters:
            is_resisterd = False
            for i in cluster_types:
                if i == cluster:
                    is_resisterd = True
            if (not (is_resisterd)) and (cluster != -1):
                cluster_types.append(cluster)

        cluster_sunspots: list[sun_data.SunspotsCluster] = []
        for cluster_type in cluster_types:
            sunspots_cluster: list[sun_data.Sunspot] = []
            for sunspot, cluster in zip(sunspots, clusters):
                if cluster == cluster_type:
                    sunspots_cluster.append(sunspot)

            x_list: list[float] = []
            y_list: list[float] = []
            for sunspot in sunspots_cluster:
                point: tuple(float, float) = sunspot.GetPoint()
                area: tuple(float, float) = sunspot.GetArea()
                x_list.append(point[0] + area[0] / 2)
                x_list.append(point[0] - area[0] / 2)
                y_list.append(point[1] + area[1] / 2)
                y_list.append(point[1] - area[1] / 2)
            upper: float = int(min(y_list))
            lower: float = int(max(y_list))
            left: float = int(min(x_list))
            right: float = int(max(x_list))

            clip_cluster: np.ndarray = sun.GetBinaryImage().copy()
            clip_cluster: np.ndarray = clip_cluster[upper:lower, left:right]
            cluster_x: float = (right - left) / 2 + left
            cluster_y: float = (lower - upper) / 2 + upper
            cluster_sunspots.append(
                sun_data.SunspotsCluster(clip_cluster, (cluster_x, cluster_y))
            )

        clusters_sunspots.append(cluster_sunspots)

    # 翌日の画像の中から同じ黒点群を探す
    for i, cluster_areas in enumerate(clusters_sunspots[:-1]):
        sun_img: np.ndarray = suns[i].GetDrawnSunpotsImage().copy()
        sun_img_next: np.ndarray = suns[i + 1].GetDrawnSunpotsImage().copy()
        slopes_rad: list[float] = []
        for cluster_area in cluster_areas:
            mse_errors: list = []
            for next_cluster_area in clusters_sunspots[i + 1]:
                mse: float = calc_mse(
                    cluster_area.GetImage(), next_cluster_area.GetImage()
                )
                mse_errors.append(mse)
            minimum_index: int = np.argmin(mse_errors)
            """
            sun_img_next = cv2.line(
                sun_img_next,
                cluster_area.GetPointInteger(),
                clusters_sunspots[i + 1][minimum_index].GetPointInteger(),
                color=(0, 0, 255),
                thickness=2,
            )
            """
            vecter: tuple = np.array(
                clusters_sunspots[i + 1][minimum_index].GetPoint()
            ) - np.array(cluster_area.GetPoint())
            slope: float = np.arctan2(vecter[1], vecter[0])
            slopes_rad.append(slope)

        rad_rotate: float = np.median(slopes_rad)
        slope: float = math.tan(rad_rotate)

        w: int = sun_img.shape[1]
        h: int = sun_img.shape[0]
        p0: tuple(int, int) = (int(-h / 2 / slope + w / 2), 0)
        p1: tuple(int, int) = (
            int((h - h / 2) / slope + w / 2),
            sun_img.shape[1],
        )

        sun_img = cv2.line(sun_img, p0, p1, color=(0, 0, 255), thickness=2)
        sun_img_next = cv2.line(
            sun_img_next, p0, p1, color=(0, 0, 255), thickness=2
        )

        # result_img: np.ndarray = cv2.hconcat([sun_img, sun_img_next])
        # cv2.imwrite("result/test.jpg", result_img)

    # 黒点の緯度経度を求める
    img_size: tuple(int, int) = suns[0].GetImageSize()
    p0 = (p0[0] / img_size[0], p0[1] / img_size[1])
    p1 = (p1[0] / img_size[0], p1[1] / img_size[1])
    for sun in suns:
        for i, sunspot in enumerate(sun.GetSunspotsNorm()):
            p: tuple(float, float) = sunspot.GetPoint()
            l: float = math.sqrt((p[0] - p0[0]) ** 2 + (p[1] - p0[1]) ** 2)
            cos_theta: float = (
                (p[0] - p0[0]) * (p1[0] - p0[0])
                + (p[1] - p0[1]) * (p1[1] - p0[1])
            ) / (
                math.sqrt((p[0] - p0[0]) ** 2 + (p[1] - p0[1]) ** 2)
                * math.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2)
            )
            theta: float = math.acos(cos_theta)
            d: float = l * math.sin(theta)

            latitude: float = math.asin(d / 0.5)
            sun.SetSunspotLatitudeRadian(i, latitude)

        sunspots_test: list[sun_data.Sunspot] = sun.GetSunspotsNorm()
        for sunspot in sunspots_test:
            lat: float = sunspot.GetLatitudeRad()
            print(math.degrees(lat))


if __name__ == "__main__":
    main()
