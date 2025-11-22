import os
import glob
import re
from typing import Dict, Tuple, Optional


def parse_raw_ranges_from_file(filepath: str) -> Dict[str, Tuple[int, int]]:
    """
    green.txt / red.txt 같은 파일에서
    '==== 객체 RGB 범위 (원값 기준) ====' 아래에 있는
    R: a ~ b / G: c ~ d / B: e ~ f 를 파싱해서
    {"R": (a,b), "G": (c,d), "B": (e,f)} 형태로 반환.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    raw_start = None
    for i, line in enumerate(lines):
        if "객체 RGB 범위 (원값 기준)" in line:
            raw_start = i
            break

    if raw_start is None:
        raise ValueError(f"'원값 기준' 구간을 찾을 수 없음: {filepath}")

    rgb_ranges: Dict[str, Tuple[int, int]] = {}

    # '원값 기준' 줄 이후부터, 공백줄이나 'margin 포함' 전까지 스캔
    pattern = re.compile(r"([RGB]):\s*([0-9.]+)\s*~\s*([0-9.]+)")

    for line in lines[raw_start + 1 : ]:
        if "객체 RGB 범위 (margin 포함)" in line:
            break
        m = pattern.search(line)
        if not m:
            continue
        ch = m.group(1)
        v1 = float(m.group(2))
        v2 = float(m.group(3))
        low, high = sorted([v1, v2])
        rgb_ranges[ch] = (low, high)

    if set(rgb_ranges.keys()) != {"R", "G", "B"}:
        raise ValueError(f"R,G,B 세 개 다 못 읽음: {filepath}, 결과={rgb_ranges}")

    return rgb_ranges


def load_color_ranges(folder: str) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """
    폴더 안의 *.txt를 전부 읽어서
    color_ranges = {
        'green': {'R': (min, max), 'G': (min, max), 'B': (min, max)},
        'red':   {...},
        ...
    }
    형태로 반환.
    """
    color_ranges: Dict[str, Dict[str, Tuple[float, float]]] = {}

    for path in glob.glob(os.path.join(folder, "*.txt")):
        label = os.path.splitext(os.path.basename(path))[0]  # green.txt -> green
        rgb_ranges = parse_raw_ranges_from_file(path)
        color_ranges[label] = rgb_ranges

    if not color_ranges:
        raise ValueError(f"폴더에 txt가 없거나 파싱 실패: {folder}")

    return color_ranges


def build_label_to_id(color_ranges: Dict[str, Dict[str, Tuple[float, float]]]) -> Dict[str, int]:
    """
    임베딩 용도로 라벨 -> 인덱스 매핑 만들어주기.
    """
    labels = sorted(color_ranges.keys())
    return {label: idx for idx, label in enumerate(labels)}


def classify_rgb(
    rgb: Tuple[float, float, float],
    color_ranges: Dict[str, Dict[str, Tuple[float, float]]],
    label_to_id: Dict[str, int],
) -> Tuple[Optional[str], Optional[int]]:
    """
    하나의 RGB 값이 들어왔을 때:
    1) 각 색 범위에 포함되는지 검사
    2) 아무 색에도 안 맞으면 (None, None)
    3) 여러 색에 동시에 포함되면 → 애매하니까 (None, None) 으로 버림
    """
    r, g, b = rgb
    candidates = []

    for label, ranges in color_ranges.items():
        r_min, r_max = ranges["R"]
        g_min, g_max = ranges["G"]
        b_min, b_max = ranges["B"]

        if (r_min <= r <= r_max) and (g_min <= g <= g_max) and (b_min <= b <= b_max):
            candidates.append(label)

    # 0개 or 2개 이상이면 None
    if len(candidates) != 1:
        return None, None

    label = candidates[0]
    return label, label_to_id[label]


def rgb_to_one_hot(
    rgb: Tuple[float, float, float],
    color_ranges: Dict[str, Dict[str, Tuple[float, float]]],
    label_to_id: Dict[str, int],
):
    """
    임베딩용으로 one-hot 벡터 만들어주는 예시.
    겹치거나 매칭 실패하면 전부 0인 벡터.
    """
    label, idx = classify_rgb(rgb, color_ranges, label_to_id)
    num_classes = len(label_to_id)
    vec = [0.0] * num_classes
    if idx is not None:
        vec[idx] = 1.0
    return label, idx, vec


if __name__ == "__main__":
    folder = "utils/color/result"

    color_ranges = load_color_ranges(folder)
    label_to_id = build_label_to_id(color_ranges)

    print("등록된 색 라벨들:", label_to_id)

    # 테스트 RGB
    test_rgb = (40.0, 100.0, 80.0)
    label, idx = classify_rgb(test_rgb, color_ranges, label_to_id)
    print(f"RGB {test_rgb} -> label={label}, id={idx}")

    label, idx, one_hot = rgb_to_one_hot(test_rgb, color_ranges, label_to_id)
    print(f"one-hot: {one_hot}")
