#!/usr/bin/env python3
"""Simple GUI label editor for reviewing the auto-label outputs.

이 툴은 auto_labels/batch_root_inference.py 로 생성된 이미지/라벨 폴더를
간단히 훑으면서 잘못 인식된 객체는 삭제하고, 누락된 객체는 추가할 수 있게
해준다. Matplotlib 인터랙션을 사용하므로 별도의 GUI 프레임워크가 필요 없다.

Usage
-----
python auto_labels/label_editor.py \
    --root ./root_dataset \
    --img-exts .jpg,.png \
    --image-dirs images,images_gt,. \
    --start-index 0

Controls
--------
* `n` / `right`  : 다음 이미지 (현재 라벨 자동 저장)
* `p` / `left`   : 이전 이미지
* `a`            : 추가 모드 토글 (센터, 첫 번째 포인트, 두 번째 포인트 순으로 3번 클릭)
* `esc`          : 추가 모드 취소
* `delete/backspace` : 선택된 라벨 삭제
* `f`            : 선택한 라벨의 앞/뒤 꼭짓점 뒤집기
* `s`            : 수동 저장
* `q`            : 프로그램 종료 (종료 시에도 변경 사항 자동 저장)
"""

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


def order_poly_ccw(poly4: np.ndarray) -> np.ndarray:
    """Return a CCW-ordered quadrilateral for stable rendering."""
    c = poly4.mean(axis=0)
    ang = np.arctan2(poly4[:, 1] - c[1], poly4[:, 0] - c[0])
    idx = np.argsort(ang)
    return poly4[idx]


def parallelogram_from_pred_triangle(tri_pred: np.ndarray) -> np.ndarray:
    """tri_pred: [cx,cy,f1x,f1y,f2x,f2y,(score?)] -> (4,2) float32(CCW)."""
    coords = np.asarray(tri_pred[:6], dtype=np.float32)
    cx, cy, x2, y2, x3, y3 = coords.tolist()
    x2m, y2m = 2 * cx - x2, 2 * cy - y2
    x3m, y3m = 2 * cx - x3, 2 * cy - y3
    poly = np.array(
        [[x2, y2], [x3, y3], [x2m, y2m], [x3m, y3m]], dtype=np.float32
    )
    return order_poly_ccw(poly)


@dataclass
class LabelEntry:
    class_id: int
    cx: float
    cy: float
    f1x: float
    f1y: float
    f2x: float
    f2y: float
    score: Optional[float] = None

    def to_line(self) -> str:
        base = (
            f"{self.class_id} "
            f"{self.cx:.2f} {self.cy:.2f} "
            f"{self.f1x:.2f} {self.f1y:.2f} "
            f"{self.f2x:.2f} {self.f2y:.2f}"
        )
        if self.score is not None:
            base += f" {self.score:.4f}"
        return base

    def to_triangle(self) -> np.ndarray:
        return np.array(
            [self.cx, self.cy, self.f1x, self.f1y, self.f2x, self.f2y],
            dtype=np.float32,
        )

    def center_point(self) -> Tuple[float, float]:
        return (self.cx, self.cy)

    def front_points(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return (self.f1x, self.f1y), (self.f2x, self.f2y)

    def rear_points(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        c0x, c0y = self.center_point()
        return (
            (2 * c0x - self.f1x, 2 * c0y - self.f1y),
            (2 * c0x - self.f2x, 2 * c0y - self.f2y),
        )

    def flip_front_back(self) -> None:
        (r1x, r1y), (r2x, r2y) = self.rear_points()
        self.f1x, self.f1y = r1x, r1y
        self.f2x, self.f2y = r2x, r2y


@dataclass
class LabelSample:
    label_path: str
    image_path: str
    dataset_dir: str


def parse_exts(exts: str) -> Tuple[str, ...]:
    cleaned = []
    for ext in (exts or "").split(","):
        e = ext.strip()
        if not e:
            continue
        if not e.startswith("."):
            e = "." + e
        cleaned.append(e.lower())
    return tuple(cleaned) if cleaned else (".jpg", ".jpeg", ".png")


def parse_image_dirs(image_dirs: str) -> Tuple[str, ...]:
    cleaned = []
    for token in (image_dirs or "").split(","):
        t = token.strip()
        if not t:
            continue
        cleaned.append(t)
    if "." not in cleaned:
        cleaned.append(".")
    return tuple(cleaned)


def load_labels(label_path: str) -> List[LabelEntry]:
    entries: List[LabelEntry] = []
    if not os.path.isfile(label_path):
        return entries
    with open(label_path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                class_id = int(float(parts[0]))
                coords = list(map(float, parts[1:7]))
                score = float(parts[7]) if len(parts) >= 8 else None
            except ValueError:
                continue
            entries.append(
                LabelEntry(
                    class_id=class_id,
                    cx=coords[0],
                    cy=coords[1],
                    f1x=coords[2],
                    f1y=coords[3],
                    f2x=coords[4],
                    f2y=coords[5],
                    score=score,
                )
            )
    return entries


def save_labels(label_path: str, entries: Sequence[LabelEntry]) -> None:
    lines = [entry.to_line() for entry in entries]
    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    with open(label_path, "w") as f:
        f.write("\n".join(lines))


def find_image_for_label(
    dataset_dir: str,
    base_name: str,
    image_dirs: Sequence[str],
    img_exts: Sequence[str],
) -> Optional[str]:
    checked = []
    for rel_dir in image_dirs:
        if rel_dir in ("", "."):
            search_dir = dataset_dir
        else:
            search_dir = os.path.join(dataset_dir, rel_dir)
        for ext in img_exts:
            candidate = os.path.join(search_dir, base_name + ext)
            checked.append(candidate)
            if os.path.isfile(candidate):
                return candidate
    return None


def collect_label_samples(
    root_dir: str, img_exts: Sequence[str], image_dirs: Sequence[str]
) -> List[LabelSample]:
    root_dir = os.path.abspath(root_dir)
    label_dirs = set()
    direct = os.path.join(root_dir, "labels")
    if os.path.isdir(direct):
        label_dirs.add(direct)

    for dirpath, dirnames, _ in os.walk(root_dir):
        if os.path.basename(dirpath) == "labels":
            label_dirs.add(dirpath)
            dirnames[:] = []

    samples: List[LabelSample] = []
    for labels_dir in sorted(label_dirs):
        dataset_dir = os.path.dirname(labels_dir)
        for name in sorted(os.listdir(labels_dir)):
            if not name.lower().endswith(".txt"):
                continue
            label_path = os.path.join(labels_dir, name)
            base_name = os.path.splitext(name)[0]
            img_path = find_image_for_label(
                dataset_dir, base_name, image_dirs, img_exts
            )
            if img_path is None:
                print(f"[WARN] 이미지 파일을 찾을 수 없습니다: {label_path}")
                continue
            samples.append(
                LabelSample(
                    label_path=label_path,
                    image_path=img_path,
                    dataset_dir=dataset_dir,
                )
            )
    return samples


class LabelEditorApp:
    def __init__(
        self,
        samples: Sequence[LabelSample],
        root_dir: str,
        start_index: int = 0,
        default_class: int = 0,
    ) -> None:
        if not samples:
            raise ValueError("라벨 파일을 찾을 수 없습니다.")
        self.samples = list(samples)
        self.root_dir = os.path.abspath(root_dir)
        self.idx = max(0, min(start_index, len(self.samples) - 1))
        self.default_class = default_class

        self.entries: List[LabelEntry] = []
        self.selected_idx: Optional[int] = None
        self.mode = "idle"
        self.add_points: List[Tuple[float, float]] = []
        self.dirty = False

        self.fig, self.ax = plt.subplots(figsize=(12, 7))
        self.ax.axis("off")
        self.fig.subplots_adjust(bottom=0.11, top=0.93)
        self.image_artist = None
        self.polygon_patches: List[Polygon] = []
        self.point_markers: List = []
        self.add_markers: List = []

        self.help_text = self.fig.text(
            0.5,
            0.015,
            "n/p(or ←/→): prev/next · a: add · esc: cancel add · del: remove · f: flip dir · s: save · q: quit",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#dddddd",
            bbox=dict(facecolor="black", alpha=0.4, pad=4),
        )
        self.status_text = self.fig.text(
            0.01,
            0.015,
            "",
            ha="left",
            va="bottom",
            fontsize=10,
            color="#ffeb3b",
            bbox=dict(facecolor="black", alpha=0.5, pad=4),
        )
        self.info_text = self.fig.text(
            0.01,
            0.985,
            "",
            ha="left",
            va="top",
            fontsize=10,
            color="#ffffff",
            bbox=dict(facecolor="black", alpha=0.4, pad=4),
        )

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("close_event", self.on_close)

    def run(self) -> None:
        self.goto(self.idx, force=True)
        plt.show()

    def on_close(self, _event) -> None:
        if self.dirty:
            try:
                save_labels(self.samples[self.idx].label_path, self.entries)
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] 라벨 저장 실패: {exc}")

    def set_status(self, text: str) -> None:
        self.status_text.set_text(text)
        self.fig.canvas.draw_idle()

    def update_info(self) -> None:
        sample = self.samples[self.idx]
        rel_label = os.path.relpath(sample.label_path, self.root_dir)
        dirty_flag = " *" if self.dirty else ""
        self.info_text.set_text(
            f"{self.idx + 1}/{len(self.samples)}{dirty_flag} · {rel_label} · "
            f"labels: {len(self.entries)}"
        )

    def refresh_patches(self) -> None:
        for patch in self.polygon_patches:
            patch.remove()
        self.polygon_patches = []
        for marker in self.point_markers:
            marker.remove()
        self.point_markers = []

        for idx, entry in enumerate(self.entries):
            poly = parallelogram_from_pred_triangle(entry.to_triangle())
            patch = Polygon(
                poly,
                closed=True,
                fill=False,
                linewidth=2.2 if idx == self.selected_idx else 1.4,
                edgecolor="#ffeb3b"
                if idx == self.selected_idx
                else "#ff5555",
            )
            self.ax.add_patch(patch)
            self.polygon_patches.append(patch)
            self.point_markers.extend(self._plot_entry_points(idx, entry))

        self.fig.canvas.draw_idle()
        self.update_info()

    def _plot_entry_points(self, idx: int, entry: LabelEntry) -> List:
        """Draw front/center keypoints to help orientation checking."""
        markers = []
        is_sel = idx == self.selected_idx
        # 중심
        cx, cy = entry.center_point()
        (center_marker,) = self.ax.plot(
            [cx],
            [cy],
            marker="o",
            linestyle="None",
            markersize=9 if is_sel else 7,
            markerfacecolor="#ffee58" if is_sel else "#fff59d",
            markeredgecolor="#000000",
            markeredgewidth=1.1,
            alpha=0.9,
        )
        markers.append(center_marker)
        # 앞쪽 두 포인트
        front_color = "#00bcd4" if is_sel else "#4dd0e1"
        for px, py in entry.front_points():
            (marker,) = self.ax.plot(
                [px],
                [py],
                marker="o",
                linestyle="None",
                markersize=8 if is_sel else 6,
                markerfacecolor=front_color,
                markeredgecolor="#00363a",
                markeredgewidth=1.0,
                alpha=0.95,
            )
            markers.append(marker)
        return markers

    def draw_add_markers(self) -> None:
        for marker in self.add_markers:
            marker.remove()
        self.add_markers = []
        colors = ["#ffeb3b", "#00e5ff", "#ff80ab"]
        for idx, (x, y) in enumerate(self.add_points):
            (marker,) = self.ax.plot(
                [x],
                [y],
                marker="o",
                color=colors[idx % len(colors)],
                markersize=8,
            )
            self.add_markers.append(marker)
        self.fig.canvas.draw_idle()

    def goto(self, new_idx: int, force: bool = False) -> None:
        if not force and (new_idx < 0 or new_idx >= len(self.samples)):
            self.set_status("더 이상 이동할 수 없습니다.")
            return
        if self.dirty:
            try:
                save_labels(self.samples[self.idx].label_path, self.entries)
                self.set_status("변경 사항 자동 저장 완료.")
            except Exception as exc:  # noqa: BLE001
                self.set_status(f"라벨 저장 실패: {exc}")
        self.idx = max(0, min(new_idx, len(self.samples) - 1))
        self.selected_idx = None
        self.mode = "idle"
        self.add_points.clear()
        self.draw_add_markers()
        self.load_current_sample()

    def load_current_sample(self) -> None:
        sample = self.samples[self.idx]
        img_bgr = cv2.imread(sample.image_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"이미지를 열 수 없습니다: {sample.image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if self.image_artist is None:
            self.image_artist = self.ax.imshow(img_rgb)
        else:
            self.image_artist.set_data(img_rgb)
        self.ax.set_title(os.path.relpath(sample.image_path, self.root_dir))
        self.entries = load_labels(sample.label_path)
        self.dirty = False
        self.refresh_patches()
        self.set_status("이미지 로드 완료.")

    def pick_entry(self, x: float, y: float, max_dist: float = 35.0) -> Optional[int]:
        best_idx = None
        best_dist = max_dist ** 2
        for idx, entry in enumerate(self.entries):
            dist = (entry.cx - x) ** 2 + (entry.cy - y) ** 2
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        return best_idx

    def on_click(self, event) -> None:
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        x, y = float(event.xdata), float(event.ydata)

        if self.mode == "add":
            self.add_points.append((x, y))
            self.draw_add_markers()
            remaining = 3 - len(self.add_points)
            if remaining <= 0:
                self.finish_add()
            else:
                self.set_status(f"추가 모드: {remaining}개의 포인트를 더 찍어주세요.")
            return

        picked = self.pick_entry(x, y)
        if picked is None:
            self.selected_idx = None
            self.refresh_patches()
            self.set_status("선택된 라벨 없음.")
            return

        self.selected_idx = picked
        self.refresh_patches()
        entry = self.entries[picked]
        sc_text = (
            f"{entry.score:.3f}" if entry.score is not None else "없음"
        )
        self.set_status(f"라벨 #{picked + 1} 선택 (score={sc_text}).")

    def finish_add(self) -> None:
        if len(self.add_points) != 3:
            return
        (cx, cy), (f1x, f1y), (f2x, f2y) = self.add_points
        new_entry = LabelEntry(
            class_id=self.default_class,
            cx=cx,
            cy=cy,
            f1x=f1x,
            f1y=f1y,
            f2x=f2x,
            f2y=f2y,
        )
        self.entries.append(new_entry)
        self.selected_idx = len(self.entries) - 1
        self.dirty = True
        self.mode = "idle"
        self.add_points.clear()
        self.draw_add_markers()
        self.refresh_patches()
        self.set_status("새 라벨이 추가되었습니다.")

    def delete_selected(self) -> None:
        if self.selected_idx is None:
            self.set_status("삭제할 라벨을 먼저 선택하세요.")
            return
        removed = self.entries.pop(self.selected_idx)
        self.selected_idx = None
        self.dirty = True
        self.refresh_patches()
        self.set_status(
            f"라벨 삭제 완료 (cx={removed.cx:.1f}, cy={removed.cy:.1f})."
        )

    def flip_selected(self) -> None:
        if self.selected_idx is None:
            self.set_status("뒤집을 라벨을 먼저 선택하세요.")
            return
        entry = self.entries[self.selected_idx]
        entry.flip_front_back()
        self.dirty = True
        self.refresh_patches()
        self.set_status("앞/뒤 꼭짓점을 뒤집었습니다.")

    def toggle_add_mode(self) -> None:
        if self.mode == "add":
            self.mode = "idle"
            self.add_points.clear()
            self.draw_add_markers()
            self.set_status("추가 모드 취소.")
        else:
            self.mode = "add"
            self.add_points.clear()
            self.draw_add_markers()
            self.set_status("추가 모드: 센터/포인트1/포인트2 순으로 클릭하세요.")

    def on_key(self, event) -> None:
        key = (event.key or "").lower()
        if key in ("n", "right"):
            self.goto(self.idx + 1)
        elif key in ("p", "left"):
            self.goto(self.idx - 1)
        elif key in ("s", "ctrl+s"):
            try:
                save_labels(self.samples[self.idx].label_path, self.entries)
                self.dirty = False
                self.update_info()
                self.set_status("수동 저장 완료.")
            except Exception as exc:  # noqa: BLE001
                self.set_status(f"라벨 저장 실패: {exc}")
        elif key in ("delete", "backspace", "d"):
            self.delete_selected()
        elif key == "f":
            self.flip_selected()
        elif key == "a":
            self.toggle_add_mode()
        elif key == "escape":
            if self.mode == "add":
                self.toggle_add_mode()
        elif key == "q":
            if self.dirty:
                try:
                    save_labels(self.samples[self.idx].label_path, self.entries)
                except Exception as exc:  # noqa: BLE001
                    print(f"[WARN] 라벨 저장 실패: {exc}")
            plt.close(self.fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-label 수정용 간단 GUI 도구"
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="labels 디렉터리를 포함하는 루트 경로",
    )
    parser.add_argument(
        "--img-exts",
        type=str,
        default=".jpg,.jpeg,.png",
        help="이미지 확장자 리스트 (콤마 구분)",
    )
    parser.add_argument(
        "--image-dirs",
        type=str,
        default="images,images_gt,.",
        help="이미지 탐색 대상 디렉터리 (dataset 폴더 기준 상대경로, 콤마 구분)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="처음 열 라벨 파일의 인덱스",
    )
    parser.add_argument(
        "--default-class",
        type=int,
        default=0,
        help="새 라벨 추가 시 사용할 클래스 ID",
    )
    args = parser.parse_args()

    img_exts = parse_exts(args.img_exts)
    image_dirs = parse_image_dirs(args.image_dirs)
    samples = collect_label_samples(args.root, img_exts, image_dirs)
    if not samples:
        raise SystemExit("[Error] 사용할 라벨 파일을 찾지 못했습니다.")

    app = LabelEditorApp(
        samples=samples,
        root_dir=args.root,
        start_index=args.start_index,
        default_class=args.default_class,
    )
    app.run()


if __name__ == "__main__":
    main()
