import torch
import mmcv
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import os.path as path

from mmcv.runner import load_checkpoint
from depth.models import build_depther
from PIL import Image
from depth.utils import colorize

cfg = mmcv.Config.fromfile(r'/home/liuchunyi/DE/configs/trap/trap_swin_l_win7_kitti_benchmark.py')
mean = [123.675, 116.28, 103.53]
std = [58.395, 57.12, 57.375]
mean = np.array(mean, dtype=np.float32)
std = np.array(std, dtype=np.float32)

# img = Image.open(r'D:\dataset\Monocular\tools\zz.jpg')
# img = np.asarray(img, dtype=np.float32)
# img = mmcv.imnormalize(img, mean, std)

# print(img)

model = build_depther(
    cfg.model,
    test_cfg=cfg.get('test_cfg'))
# torch.nn.Module.named_modules()
# for n, m in model.named_modules():
#     print(n)
# load_checkpoint(model, r'/home/liuchunyi/DE/log/kitti_l_swin_win7_benchmark2/best_abs_rel_iter_174400.pth', map_location='cpu')
# load_checkpoint(model, r'/home/liuchunyi/DE/log/kitti_l_swin_win12_benchmark_online53/best_abs_rel_iter_5600.pth', map_location='cpu')
load_checkpoint(model, r'/home/liuchunyi/DE/log/kitti_l_swin_win12_benchmark_online53/best_abs_rel_iter_1600.pth', map_location='cpu')

img_metas = {'flip':False,
             'img_shape':(352, 1216, 3),
             "ori_shape":(352, 1216, 3),
             "pad_shape":(352, 1216, 3),
             # 'cam_intrinsic':[[7.188560e+02, 0.000000e+00, 6.071928e+02, 4.538225e+01],
             #                [0.000000e+00, 7.188560e+02, 1.852157e+02, -1.130887e-01],
             #                [0.000000e+00, 0.000000e+00, 1.000000e+00, 3.779761e-03]],
             'scale_factor': np.array([1, 1, 1, 1], dtype=np.float32)}

root = '/home/liuchunyi/DE/data/kitti/official/depth_selection/test_depth_prediction_anonymous/image'
cam_intrinsic = '/home/liuchunyi/DE/data/kitti/official/depth_selection/test_depth_prediction_anonymous/intrinsics'
dest = '/home/liuchunyi/DE/BBB_A'
dirlist = os.listdir(root)
print(len(dirlist))
with torch.no_grad():
    model = model.eval().cuda()
    for p in dirlist:
        img = Image.open(path.join(root, p))
        ins_p = p.replace('.png', '.txt')
        with open(path.join(cam_intrinsic, ins_p), mode='r') as cf:
            ins = cf.readline().split(' ')[:9]
        img_metas['cam_intrinsic'] = np.array(ins).reshape((3, 3))
        img = np.asarray(img, dtype=np.float32)
        img = mmcv.imnormalize(img, mean, std)
        result = model([torch.Tensor(img).permute(2, 0, 1).unsqueeze(0).cuda()], [[img_metas]],
                       return_loss=False)
        depth = result[0]
        depth = np.array(depth.squeeze() * 256, dtype=np.uint16)
        mmcv.imwrite(depth, path.join(dest, p))

# plt.figure()
# plt.axis("off")
# # plt.imshow(result[0][0])
# plt.imshow(depth[0])
# plt.show()


