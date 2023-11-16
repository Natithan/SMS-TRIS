all_factors = [
    'paddle_left_y',
    'paddle_right_y',
    'ball_x',
    'ball_y',
    'ball_vel_dir',
    'ball_vel_magn',
    'score_left',
    'score_right'
]
idx = 0
# direc = f"/cw/liir_code/NoCsBack/nathan/p2/CITRIS/data_generation/ood_pong/100_train_points/{all_factors[idx]}/"
direc = f"/cw/liir_code/NoCsBack/nathan/p2/CITRIS/data_generation/ood_pong/100_train_points_no_intvs_no_noise/{all_factors[idx]}/"
from util import plot_tensor_as_vid
import numpy as np
f = np.load(direc + 'train.npz')
imgs = f['images']
plot_tensor_as_vid(imgs, pause_time = .2)