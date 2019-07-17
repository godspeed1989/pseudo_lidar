
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

import preprocessing.kitti_util as kitti_util
from view_pc_bin import view

def project_disp_to_depth(calib, disp, seg_mask=None, max_high=1):
    disp[disp < 0] = 0
    baseline = 1.35
    mask = disp > 0
    if seg_mask is not None:
        mask = np.logical_and(mask, seg_mask)
    #
    #depth = calib.f_u * baseline / (disp + 1. - mask)
    depth = baseline / (disp + 1. - mask)
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth])
    points = points.reshape((3, -1))
    points = points.T
    points = points[mask.reshape(-1)]
    cloud = calib.project_image_to_velo(points)
    valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
    return cloud[valid]

def to_bin_file(lidar, predix, save_dir='./'):
    lidar = lidar.astype(np.float32)
    path = os.path.join(save_dir, '{}.bin'.format(save_dir, predix))
    lidar.tofile(path)
    return path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage ./exe 000025')
        sys.exit(0)

    img_file = sys.argv[1] + '.png'
    calib_file = sys.argv[1] + '.txt'
    depth_file = sys.argv[1] + 'd.png'
    seg_file = sys.argv[1] + 's.png'
    gt_lidar_file = sys.argv[1] + 'gt.bin'

    image = cv2.imread(img_file)
    calib = kitti_util.Calibration(calib_file)
    depth_map = cv2.imread(depth_file, 0) / 256.
    seg_map = cv2.imread(seg_file, 0)
    seg_mask = (seg_map==13).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)) # 椭圆
    seg_mask = cv2.erode(seg_mask, kernel) > 0

    gt_lidar = np.fromfile(gt_lidar_file, dtype=np.float32).reshape(-1, 4)
    proj_lidar = project_disp_to_depth(calib, depth_map, seg_mask)
    # pad 1 in the indensity dimension
    proj_lidar = np.concatenate([proj_lidar, np.ones((proj_lidar.shape[0], 1))], 1)

    out_lidar = np.concatenate([gt_lidar, proj_lidar], axis=0)

    f = to_bin_file(out_lidar, sys.argv[1]+'p')
    print('Finish Depth {} {}'.format(sys.argv[1], out_lidar.shape[0]))
    view(f)