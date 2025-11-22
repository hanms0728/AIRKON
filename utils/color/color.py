import cv2
import numpy as np
import glob
import os

def get_center_mean_color(img_rgb, crop_ratio=0.5):
    """
    이미지 중앙 부분만 잘라서 평균 RGB 색을 반환.
    crop_ratio: 0~1 사이, 0.5면 가로/세로의 50% 영역만 중앙에서 사용.
    """
    h, w, _ = img_rgb.shape
    ch = int(h * crop_ratio)
    cw = int(w * crop_ratio)
    
    y1 = (h - ch) // 2
    y2 = y1 + ch
    x1 = (w - cw) // 2
    x2 = x1 + cw
    
    center_crop = img_rgb[y1:y2, x1:x2]
    mean_color = center_crop.reshape(-1, 3).mean(axis=0)  # [R, G, B]
    return mean_color

def extract_object_color_range(folder_path, crop_ratio=0.5, margin=10):
    # 이미지 확장자들
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
    
    if not image_paths:
        print("이미지가 없습니다. 폴더 경로를 확인해줘.")
        return None
    
    colors = []  # 각 이미지에서 얻은 [R, G, B]
    
    for img_path in image_paths:
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"이미지를 읽을 수 없음: {img_path}")
            continue
        
        # BGR -> RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # 중앙 영역 평균색 추출
        mean_color = get_center_mean_color(img_rgb, crop_ratio=crop_ratio)
        colors.append(mean_color)
        
        print(f"{os.path.basename(img_path)} -> mean RGB: {mean_color.astype(int)}")
    
    if not colors:
        print("유효한 이미지에서 색을 추출하지 못했어.")
        return None
    
    colors = np.array(colors)  # shape: (N, 3)
    
    # 채널별 최소/최대
    rgb_min = colors.min(axis=0)  # [R_min, G_min, B_min]
    rgb_max = colors.max(axis=0)  # [R_max, G_max, B_max]
    
    # margin 적용 (0~255 클램프)
    rgb_min_margin = np.clip(rgb_min - margin, 0, 255)
    rgb_max_margin = np.clip(rgb_max + margin, 0, 255)
    
    print("\n==== 객체 RGB 범위 (원값 기준) ====")
    print(f"R: {rgb_min[0]:.1f} ~ {rgb_max[0]:.1f}")
    print(f"G: {rgb_min[1]:.1f} ~ {rgb_max[1]:.1f}")
    print(f"B: {rgb_min[2]:.1f} ~ {rgb_max[2]:.1f}")
    
    print("\n==== 객체 RGB 범위 (margin 포함, threshold용) ====")
    print(f"R: {rgb_min_margin[0]:.0f} ~ {rgb_max_margin[0]:.0f}")
    print(f"G: {rgb_min_margin[1]:.0f} ~ {rgb_max_margin[1]:.0f}")
    print(f"B: {rgb_min_margin[2]:.0f} ~ {rgb_max_margin[2]:.0f}")
    
    result = {
        "per_image_colors": colors,
        "rgb_min": rgb_min,
        "rgb_max": rgb_max,
        "rgb_min_margin": rgb_min_margin,
        "rgb_max_margin": rgb_max_margin,
    }
    return result

if __name__ == "__main__":
    # 여기에 네 이미지 폴더 경로 넣으면 됨
    folder = "/path/to/your/image_folder"  # <- 바꿔 줘
    extract_object_color_range(folder, crop_ratio=0.5, margin=10)
