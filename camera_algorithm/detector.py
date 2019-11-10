import cv2
import numpy as np
import random
import sys

def main(source,base):

    # 対象画像を指定
    base_image_path = source
    temp_image_path = base

    # 画像をグレースケールで読み込み
    gray_base_src = cv2.imread(base_image_path, 0)
    gray_temp_src = cv2.imread(temp_image_path, 0)

    # マッチング結果書き出し準備
    # 画像をBGRカラーで読み込み
    color_base_src = cv2.imread(base_image_path, 1)
    color_temp_src = cv2.imread(temp_image_path, 1)

    # 特徴点の検出
    type = cv2.AKAZE_create()
    kp_01, des_01 = type.detectAndCompute(gray_base_src, None)
    kp_02, des_02 = type.detectAndCompute(gray_temp_src, None)

    # マッチング処理
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    try:
        matches = bf.match(des_01, des_02)
        matches = sorted(matches, key = lambda x:x.distance)
        mutch_image_src = cv2.drawMatches(color_base_src, kp_01, color_temp_src, kp_02, matches[:10], None, flags=2)


        dist = [m.distance for m in matches]
        ret = sum(dist) / len(dist)
        print(ret)
        return ret
    except :
        print("E")
        return 0
    # 結果の表示
    cv2.imshow("mutch_image_src", mutch_image_src)
    cv2.imshow("02_result08", mutch_image_src)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

main(sys.argv[1],sys.argv[2])