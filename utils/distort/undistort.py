import numpy as np
import cv2
import argparse

def build_scaled_K(K, calib_w, calib_h, new_w, new_h):
    sx, sy = new_w / calib_w, new_h / calib_h
    K2 = K.copy().astype(np.float64)
    K2[0,0] *= sx  # fx
    K2[1,1] *= sy  # fy
    K2[0,2] *= sx  # cx
    K2[1,2] *= sy  # cy
    return K2

def build_dist_only_K(new_w, new_h, focal_ratio=0.9):
    f = focal_ratio * max(new_w, new_h)
    return np.array([[f, 0, new_w/2],
                     [0, f, new_h/2],
                     [0, 0, 1]], dtype=np.float64)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="cam2_calibration_results.npz")
    ap.add_argument("--img", required=True, help="보정할 이미지 경로")
    ap.add_argument("--mode", choices=["scaled-k", "dist-only"], default="scaled-k")
    ap.add_argument("--focal_ratio", type=float, default=0.9, help="dist-only 모드용 f 비율")
    ap.add_argument("--out", default="cam1_undistorted_image.jpg")
    args = ap.parse_args()

    data = np.load(args.calib)
    dist = data["dist"].astype(np.float64)
    img = cv2.imread(args.img)
    if img is None:
        print("이미지 로드 실패:", args.img); return
    h, w = img.shape[:2]

    if args.mode == "scaled-k":
        if "K" not in data or "calib_w" not in data or "calib_h" not in data:
            raise RuntimeError("scaled-k 모드에는 K, calib_w, calib_h가 저장돼 있어야 합니다.")
        K = data["K"].astype(np.float64)
        K_use = build_scaled_K(K, int(data["calib_w"]), int(data["calib_h"]), w, h)
    else:
        # dist만 사용하여 임시 K 구성
        K_use = build_dist_only_K(w, h, focal_ratio=args.focal_ratio)

        # --- 로드 직후 Debug 출력 ---
    print("[DBG] calib_w,h:", int(data["calib_w"]), int(data["calib_h"]))
    print("[DBG] target_w,h:", w, h)
    print("[DBG] K (loaded):\n", data["K"])
    print("[DBG] dist:", dist.ravel())

    
    K_use = data["K"].astype(np.float64)
    newK, _ = cv2.getOptimalNewCameraMatrix(K_use, dist, (w, h), alpha=0.0)  # alpha 0~1 시험
    print("[DBG] newK:\n", newK)

    map1, map2 = cv2.initUndistortRectifyMap(K_use, dist, None, newK, (w, h), cv2.CV_16SC2)
    und = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
    cv2.imwrite("und_remap2.jpg", und)
if __name__ == "__main__":
    main()
