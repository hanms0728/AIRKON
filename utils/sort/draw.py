import os
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
'''
기능: SORT 알고리즘이 저장한 tracking_output.txt 파일을 불러와서 트랙 ID별로 색상을 달리하여 시각화
출력: /tracking_visualizations/ 폴더에 프레임별 시각화
'''
def obb_to_quad(cx, cy, l, w, yaw_deg):
    yaw = math.radians(yaw_deg)
    dx, dy = l / 2.0, w / 2.0
    corners = np.array([[ dx,  dy],
                        [ dx, -dy],
                        [-dx, -dy],
                        [-dx,  dy]])
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s],
                  [s,  c]])
    return corners @ R.T + np.array([cx, cy])

def color_for_track(track_id, cmap_name="tab20"):
    cmap = plt.get_cmap(cmap_name)
    return cmap((track_id % 20) / 20.0)

def load_tracks_txt(txt_path):
    by_frame = defaultdict(list)
    
    with open(txt_path, "r", encoding="utf-8") as f:
        header_line = f.readline().strip()
        if header_line.startswith('#'):
            header_line = header_line[1:].strip()
        
        fieldnames = [h.strip() for h in header_line.split(',')]
        f.seek(0)
        for line in f:
            if line.startswith('#'):
                continue
            break
        
        reader = csv.DictReader(f, fieldnames=fieldnames, delimiter=',')
        
        for row in reader:
            if not row or not row.get("frame_id"):
                continue
                
            try:
                fr = int(row["frame_id"])
                tid = int(row["track_id"])
                x = float(row["x_center"])
                y = float(row["y_center"])
                l = float(row["length"])
                w = float(row["width"])
                th = float(row["angle"])
                
                st = "Confirmed" 
                
                by_frame[fr].append({
                    "id": tid, "cx": x, "cy": y,
                    "l": l, "w": w, "yaw": th, "state": st
                })
            except Exception as e:
                continue
                
    frames = sorted(by_frame.keys())
    return frames, by_frame

def draw_track_box(ax, box, color, linewidth=1.5):
    q = obb_to_quad(box["cx"], box["cy"], box["l"], box["w"], box["yaw"])
    q = np.vstack([q, q[0]])  
    ax.plot(q[:, 0], q[:, 1], color=color, linewidth=linewidth, alpha=0.95, zorder=3)

def draw_track_id(ax, box, tid, color):
    ax.text(box["cx"], box["cy"],
            str(tid),
            fontsize=9, ha="center", va="center",
            color=color, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6),
            zorder=4)

def visualize_tracks(txt_path,
                     out_dir,
                     xlim=(-120, -30),
                     ylim=(40, -80),
                     bg_path=None,
                     bg_alpha=0.6,
                     bg_pixels=(690, 920),
                     dpi=110,
                     only_confirmed=True,
                     trail_len=20,
                     draw_trail=True,
                     trail_linewidth=1.5):

    os.makedirs(out_dir, exist_ok=True)
    
    frames, by_frame = load_tracks_txt(txt_path) 
    
    if not frames:
        print(f"오류: {txt_path} 파일에서 추적 데이터를 찾을 수 없습니다.")
        return

    trails = defaultdict(list)

    fig_w = bg_pixels[0] / float(dpi)
    fig_h = bg_pixels[1] / float(dpi)

    for fr in tqdm(frames, desc="Rendering frames"):
        boxes = by_frame[fr]

        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        ax.set_aspect("equal", "box")
        ax.set_xlim(*xlim)
        ax.set_ylim(ylim[0], ylim[1]) 
        ax.grid(True, ls="--", alpha=0.35)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"Tracks @ frame {fr}")

        # 배경(선택)
        if bg_path and os.path.exists(bg_path):
            try:
                bg_img = plt.imread(bg_path)
                y_min_extent, y_max_extent = min(ylim[0], ylim[1]), max(ylim[0], ylim[1])
                extent = (xlim[0], xlim[1], y_min_extent, y_max_extent)
                ax.imshow(bg_img, extent=extent, origin="lower", alpha=bg_alpha, zorder=0)
            except Exception:
                pass

        for b in boxes:
            if only_confirmed and b["state"] != "Confirmed":
                continue
            
            tid = b["id"]
            color = color_for_track(tid)

            trails[tid].append((b["cx"], b["cy"]))
            if len(trails[tid]) > trail_len:
                trails[tid] = trails[tid][-trail_len:]

            if draw_trail and len(trails[tid]) >= 2:
                xs, ys = zip(*trails[tid])
                ax.plot(xs, ys, linewidth=trail_linewidth, alpha=0.9, color=color, zorder=2)

            draw_track_box(ax, b, color=color, linewidth=1.8)
            draw_track_id(ax, b, tid, color=color)

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"tracks_{fr:06d}.png"), dpi=200)
        plt.close(fig)
        
    print(f"\n✅ 시각화 완료. 결과 이미지가 '{out_dir}' 폴더에 저장되었습니다.")

if __name__ == "__main__":
    TXT_PATH = "tracking_output.txt" 
    OUT_DIR = "tracking_visualizations"
    BG_PATH = "칼라맵.png"

    visualize_tracks(
        txt_path=TXT_PATH,
        out_dir=OUT_DIR,
        xlim=(-120, -30), 
        ylim=(40, -80),  
        bg_path=BG_PATH,
        bg_alpha=0.6,
        bg_pixels=(690, 920),
        dpi=110,
        only_confirmed=True,
        trail_len=25,
        draw_trail=True,
        trail_linewidth=1.5
    )