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



def build_color_rules_dict(folder: str) -> dict:
    """
    폴더 안의 txt 파일들을 읽어
    _COLOR_RGB_RULES 형태의 dict로 변환해주는 함수.
    
    return 예시:
    {
        "red":    {"min": (0.21, 0.08, 0.06), "max": (0.58, 0.46, 0.49)},
        "green":  {"min": (...), "max": (...)},
        ...
    }
    """
    color_ranges = load_color_ranges(folder)  # 원값 기준 RGB 범위 (0~255)
    result = {}

    for label, ranges in color_ranges.items():
        r_min, r_max = ranges["R"]
        g_min, g_max = ranges["G"]
        b_min, b_max = ranges["B"]

        # 0~1 normalization
        min_tuple = (r_min / 255.0, g_min / 255.0, b_min / 255.0)
        max_tuple = (r_max / 255.0, g_max / 255.0, b_max / 255.0)

        # 원본 0~255
        _min_tuple = (r_min, g_min , b_min )
        _max_tuple = (r_max , g_max,  b_max )

        result[label] = {
            "min": tuple(round(v, 4) for v in min_tuple),
            "max": tuple(round(v, 4) for v in max_tuple),
            "_min": tuple(round(v, 4) for v in _min_tuple),
            "_max": tuple(round(v, 4) for v in _max_tuple),
        }

    return result


def print_color_rules(folder: str):
    """
    _COLOR_RGB_RULES = {...} 형태로 깔끔히 출력
    """
    rules = build_color_rules_dict(folder)

    print("_COLOR_RGB_RULES = {")
    for label, val in rules.items():
        print(f'    "{label}": {{')
        print(f"        \"min\": {val['min']},")
        print(f"        \"max\": {val['max']},")
        print("    },")
    print("}")
    print("====================")
    print("_COLOR_RGB_RULES_0_255 = {")
    for label, val in rules.items():
        print(f'    "{label}": {{')
        print(f"        \"min\": {val['_min']},")
        print(f"        \"max\": {val['_max']},")
        print("    },")
    print("}")

folder = "utils/color/result"
print_color_rules(folder)