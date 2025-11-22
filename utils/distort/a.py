import numpy as np
import cv2
import glob

CHECKERBOARD_SIZE = (13, 9)
SQUARE_SIZE = 40.0
CALIB_IMAGES_PATH = "cam_28/*.jpg"
images = glob.glob(CALIB_IMAGES_PATH)
if not images:
    print(f"오류: {CALIB_IMAGES_PATH} 경로에 이미지가 없습니다.")
    raise SystemExit

objpoints, imgpoints = [], []
objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE

print(f"{len(images)}장의 이미지를 처리합니다...")
gray = None
for fname in images:
    img = cv2.imread(fname)
    if img is None: 
        print("읽기 실패:", fname); 
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SIZE, None)
    if not ret:
        print("코너 실패:", fname); 
        continue

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    objpoints.append(objp.copy())
    imgpoints.append(corners2)

if not objpoints:
    print("유효한 코너가 없습니다.")
    raise SystemExit

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

if not ret:
    print("❌ 캘리브레이션 실패")
    raise SystemExit

h_calib, w_calib = gray.shape[:2]
print("\n✅ 캘리브레이션 성공")
print("K=\n", K)
print("dist=\n", dist.ravel())

# 저장은 dist 중심 + 기준 해상도도 함께 저장(스케일링용)
np.savez(
    "cam28_calibration_results.npz",
    dist=dist,
    K=K,                 # 옵션: scaled-k 모드에서만 사용
    calib_w=w_calib,
    calib_h=h_calib
)
print("➡ cam1_calibration_results.npz 저장 완료")
