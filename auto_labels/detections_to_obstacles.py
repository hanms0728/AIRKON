'''
Script to read detections as 2D parallelograms, defined by center point and front corners (pixel coordinates)
and convert them into 3D bounding boxes (center point and heading angle / rotation / yaw) given a gnss_ref reference point in TAF.

$ python -f detections.csv -c k729_cam1 > obstacles.csv
'''

import argparse
import os
import sys
import numpy as np
import pandas as pd

from geometry import complete_parallelograms, get_heading
from projection import pixel_to_world_plane, rescale_pixels
from calibration import H

input_image_size = (1440, 810)

def main(args):
    rescale = lambda x: x
    if args.homography_size != input_image_size:
        rescale = lambda x: rescale_pixels(x, src=input_image_size, target=args.homography_size)

    # 1. Read ground plate parallelogram points
    df = pd.read_csv(args.file)
    df = df.sort_values('ts')

    front_1 = np.dstack([df['f1x'], df['f1y']]).reshape(-1, 2)  # (n, 2)
    front_2 = np.dstack([df['f2x'], df['f2y']]).reshape(-1, 2)  # (n, 2)
    centers = np.dstack([df['cx'], df['cy']]).reshape(-1, 2)  # (n, 2)

    # 2. Project to world coordinates
    front_1_p = pixel_to_world_plane(front_1, H[args.camera], rescale=rescale)  # (n, 2)
    front_2_p = pixel_to_world_plane(front_2, H[args.camera], rescale=rescale)  # (n, 2)
    centers_p = pixel_to_world_plane(centers, H[args.camera], rescale=rescale)  # (n, 2)

    # 3. Infer back corners
    corners_p = complete_parallelograms(front_1_p, front_2_p, centers_p)  # (n, 5, 2)

    # 4. Calculate heading angles
    headings_p = get_heading(corners_p[:, 0], corners_p[:, 2], False)

    # 5. Construct obstacle tuples with given fixed size
    df_out = pd.DataFrame({'ts': df['ts'], 'cx': corners_p[:, -1, 0], 'cy': corners_p[:, -1, 1], 'yaw': headings_p})
    df_out.to_csv(sys.stdout, header=True, index=None)


def try_extract_ts(path: str) -> float:
    '''Tries to parse time stamp from file name like "k733_cam1_1731676324-676637912.csv"'''
    file_name = os.path.basename(path)
    parts = file_name.split('_')
    if len(parts) > 1:
        subparts = parts[-1].split('.')[0].split('-')
        if len(subparts) == 2:
            return float(subparts[0]) + float(subparts[1]) // 1e9
    return 0.


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert detections to obstacles')
    parser.add_argument('-f', '--file', required=True, help='Path to detections CSV file')
    parser.add_argument('-c', '--camera', choices=H.keys(), default=list(H.keys())[0], help='Camera name')
    parser.add_argument('--homography_size', required=False, nargs=2, type=int, default=(1920, 1080), help='Size (width, height) of image used for homography calculation')
    args = parser.parse_args()
    
    args.homography_size = tuple(args.homography_size)

    main(args)
