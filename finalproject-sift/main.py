import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_image(path: str):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Gagal membaca gambar: {path}")
    return img

def save_rgb_image(path_out: str, bgr_img: np.ndarray):
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 6))
    plt.axis("off")
    plt.imshow(rgb)
    plt.tight_layout()
    plt.savefig(path_out, dpi=200, bbox_inches="tight")
    plt.close()

def main():
    # ====== PATH ======
    img1_path = os.path.join("dataset", "img1.jpg")
    img2_path = os.path.join("dataset", "img2.jpg")
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)

    # ====== LOAD ======
    img1 = read_image(img1_path)
    img2 = read_image(img2_path)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # ====== SIFT ======
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    print(f"Jumlah keypoints img1: {len(kp1)}")
    print(f"Jumlah keypoints img2: {len(kp2)}")

    if des1 is None or des2 is None:
        raise ValueError("Descriptor None. Coba pakai gambar yang lebih tajam/bertekstur, atau resolusi lebih besar.")

    # ====== VISUALIZE KEYPOINTS ======
    kp_img1 = cv2.drawKeypoints(img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    kp_img2 = cv2.drawKeypoints(img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    save_rgb_image(os.path.join(out_dir, "keypoints_img1.png"), kp_img1)
    save_rgb_image(os.path.join(out_dir, "keypoints_img2.png"), kp_img2)

    # ====== MATCHING
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn_matches = bf.knnMatch(des1, des2, k=2)

    good = []
    ratio = 0.75 
    for m, n in knn_matches:
        if m.distance < ratio * n.distance:
            good.append(m)

    print(f"Total matches (knn): {len(knn_matches)}")
    print(f"Good matches (ratio test={ratio}): {len(good)}")

    # gambar top-N good matches
    top_n = min(50, len(good))
    match_vis = cv2.drawMatches(
        img1, kp1, img2, kp2,
        sorted(good, key=lambda x: x.distance)[:top_n],
        None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    save_rgb_image(os.path.join(out_dir, "matches_top.png"), match_vis)


    if len(good) >= 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is not None:
            inliers = int(mask.sum())
            print(f"Homography berhasil. Inliers: {inliers}/{len(good)}")

            h1, w1 = img1.shape[:2]
            corners = np.float32([[0,0],[w1,0],[w1,h1],[0,h1]]).reshape(-1,1,2)
            proj = cv2.perspectiveTransform(corners, H)

            img2_box = img2.copy()
            img2_box = cv2.polylines(img2_box, [np.int32(proj)], True, (0,255,0), 3, cv2.LINE_AA)
            save_rgb_image(os.path.join(out_dir, "homography_box.png"), img2_box)
        else:
            print("Homography gagal (H None).")
    else:
        print("Good matches kurang dari 10, homography dilewati.")

    print(f"Selesai. Output tersimpan di folder: {out_dir}/")

if __name__ == "__main__":
    main()

