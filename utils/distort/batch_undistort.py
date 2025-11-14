import os
import argparse
import cv2
import numpy as np

def build_scaled_K(K, calib_w, calib_h, new_w, new_h):
    sx, sy = new_w / calib_w, new_h / calib_h
    K2 = K.copy().astype(np.float64)
    K2[0, 0] *= sx  # fx
    K2[1, 1] *= sy  # fy
    K2[0, 2] *= sx  # cx
    K2[1, 2] *= sy  # cy
    return K2

def build_dist_only_K(new_w, new_h, focal_ratio=0.9):
    f = focal_ratio * max(new_w, new_h)
    return np.array([[f, 0, new_w / 2],
                     [0, f, new_h / 2],
                     [0, 0, 1]], dtype=np.float64)

def undistort_image(img, data, mode="scaled-k", focal_ratio=0.9, dbg_prefix=""):
    """
    네가 준 단일 이미지용 코드와 거의 동일하게 동작하도록 만든 함수.
    """
    dist = data["dist"].astype(np.float64)
    h, w = img.shape[:2]

    # --- 원래 코드의 mode 분기 그대로 유지 ---
    if mode == "scaled-k":
        if "K" not in data or "size" not in data:
            raise RuntimeError("scaled-k 모드에는 K, size가 저장돼 있어야 합니다.")
        K = data["K"].astype(np.float64)
        calib_w = int(data["size"][0])
        calib_h = int(data["size"][1])
        K_use = build_scaled_K(K, calib_w, calib_h, w, h)
    else:
        K_use = build_dist_only_K(w, h, focal_ratio=focal_ratio)

    # --- Debug 출력 (원래 코드와 비슷하게) ---
    if "size" in data:
        print(f"{dbg_prefix}[DBG] calib_w,h:", int(data["size"][0]), int(data["size"][1]))
    print(f"{dbg_prefix}[DBG] target_w,h:", w, h)
    if "K" in data:
        print(f"{dbg_prefix}[DBG] K (loaded):\n", data["K"])
    print(f"{dbg_prefix}[DBG] dist:", dist.ravel())

    # ☆ 여기서 원래 코드처럼 K_use를 data["K"]로 덮어씀
    K_use = data["K"].astype(np.float64)

    # getOptimalNewCameraMatrix도 그대로 사용
    newK, _ = cv2.getOptimalNewCameraMatrix(K_use, dist, (w, h), alpha=1.0)
    print(f"{dbg_prefix}[DBG] newK:\n", newK)

    map1, map2 = cv2.initUndistortRectifyMap(
        K_use, dist, None, newK, (w, h), cv2.CV_16SC2
    )
    und = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
    return und

def is_image_file(name):
    lower = name.lower()
    return lower.endswith(".jpg") or lower.endswith(".jpeg") \
        or lower.endswith(".png") or lower.endswith(".bmp") \
        or lower.endswith(".tif") or lower.endswith(".tiff")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="cam2_calibration_results.npz")
    ap.add_argument("--root", required=True, help="원본 이미지들이 들어있는 루트 폴더")
    ap.add_argument("--mode", choices=["scaled-k", "dist-only"], default="scaled-k")
    ap.add_argument("--focal_ratio", type=float, default=0.9, help="dist-only 모드용 f 비율")
    ap.add_argument("--suffix", default="_undist", help="루트 폴더 뒤에 붙일 접미사")
    args = ap.parse_args()

    root_dir = os.path.abspath(args.root)
    if not os.path.isdir(root_dir):
        print("루트 폴더가 아닙니다:", root_dir)
        return

    data = np.load(args.calib)

    # root_undist 폴더 경로
    parent_dir = os.path.dirname(root_dir)
    root_name = os.path.basename(root_dir.rstrip("/\\"))
    dst_root = os.path.join(parent_dir, root_name + args.suffix)
    os.makedirs(dst_root, exist_ok=True)
    print("[INFO] 출력 루트 폴더:", dst_root)

    # 전체 구조를 그대로 복사하면서 이미지만 undistort
    for dirpath, dirnames, filenames in os.walk(root_dir):
        rel = os.path.relpath(dirpath, root_dir)   # root 기준 상대 경로
        dst_dir = os.path.join(dst_root, rel)
        os.makedirs(dst_dir, exist_ok=True)

        for fname in filenames:
            if not is_image_file(fname):
                continue

            src_path = os.path.join(dirpath, fname)
            dst_path = os.path.join(dst_dir, fname)

            img = cv2.imread(src_path)
            if img is None:
                print("[WARN] 이미지 로드 실패, 건너뜀:", src_path)
                continue

            dbg_prefix = f"[{os.path.join(rel, fname)}] "
            print(f"[INFO] 처리 중: {src_path} -> {dst_path}")
            und = undistort_image(
                img,
                data,
                mode=args.mode,
                focal_ratio=args.focal_ratio,
                dbg_prefix=dbg_prefix
            )

            cv2.imwrite(dst_path, und)

    print("[INFO] 전체 처리 완료")

if __name__ == "__main__":
    main()