from os.path import join as jn
DATASET_ROOT_FOLDER = "/cw/liir_data/NoCsBack/"
# RP_DATASET_PATH = j(DATASET_ROOT_FOLDER,'bair_robot_pushing')
RP_DATASET_PATH = jn(DATASET_ROOT_FOLDER, 'bair_robot_pushing_small')
CLEVRER_ROOT = jn(DATASET_ROOT_FOLDER, 'CLEVRER')
TRIS_ROOT = jn(DATASET_ROOT_FOLDER, 'TRIS')
TC3DI_ROOT = jn(TRIS_ROOT, "causal3d_time_dep_all7_conthue_01_coarse") # /cw/liir_data/NoCsBack/TRIS/causal3d_time_dep_all7_conthue_01_coarse
OLD_PONG_ROOT = jn(TRIS_ROOT, "complexpong_time_dep_015_32width_singleint")
P2_ROOT = '/cw/liir_code/NoCsBack/nathan/p2'
VIDVQVAE_ROOT = jn(P2_ROOT, 'Video-VQVAE-main')
CITRIS_ROOT = jn(P2_ROOT, 'CITRIS') # /cw/liir_code/NoCsBack/nathan/p2/CITRIS
CITRIS_CKPT_ROOT = jn(CITRIS_ROOT, 'experiments', 'checkpoints') # /cw/liir_code/NoCsBack/nathan/p2/CITRIS/experiments/checkpoints
# # get hostname
# import socket
# hostname = socket.gethostname()
# CITRIS_CKPT_ROOT =
PONG_ROOT = jn(CITRIS_ROOT, "data_generation/id_pong/250000_train_points") # /cw/liir_code/NoCsBack/nathan/p2/CITRIS/data_generation/id_pong/250000_train_points
PONG_OOD_ROOT = jn(CITRIS_ROOT, "data_generation/ood_pong/10000_train_points") # /cw/liir_code/NoCsBack/nathan/p2/CITRIS/data_generation/ood_pong/10000_train_points
PSPONG_ROOT = jn(CITRIS_ROOT, "data_generation/id_pong/100000_train_points_paper_settings") # /cw/liir_code/NoCsBack/nathan/p2/CITRIS/data_generation/id_pong/100000_train_points_paper_settings
PSPONG_OOD_ROOT = jn(CITRIS_ROOT, "data_generation/ood_pong/10000_train_points_paper_settings") # /cw/liir_code/NoCsBack/nathan/p2/CITRIS/data_generation/ood_pong/10000_train_points_paper_settings
# TC3DI_OOD_COMMON_ROOT = jn(CITRIS_ROOT, "data_generation/temporal_causal3dident")
TC3DI_OOD_COMMON_ROOT = jn(TC3DI_ROOT, "ood")
SMALL_TC3DI_OOD_ROOT = jn(TC3DI_OOD_COMMON_ROOT, "1000_points")
TC3DI_OOD_ROOT = jn(TC3DI_OOD_COMMON_ROOT, "100000_points") # /cw/liir_data/NoCsBack/TRIS/causal3d_time_dep_all7_conthue_01_coarse/ood/100000_points
LATEX_IMG_FOLDER = jn(P2_ROOT, 'latex', 'img')
DA_LOGS_ROOT = jn(CITRIS_ROOT, 'DA_logs')
BLENDER_PATH = jn(P2_ROOT, "blender-2.93.8-linux-x64", "blender")
IMG_GEN_SCRIPT_PATH = jn(CITRIS_ROOT, "data_generation", "temporal_causal3dident",
                                  "generate_causal3dident_images.py")
AE_SHAPES_CKPTS = [jn(CITRIS_CKPT_ROOT, specific_path) for specific_path in [
    "56np5iqb_Autoencoder_shapes/epoch=999-step=479000.ckpt",
    "bge45u9h_Autoencoder_shapes/epoch=999-step=479000.ckpt",
    "1r2g0lxv/epoch=995-step=486048.ckpt",
    "utlrlfb8_Autoencoder_shapes/epoch=999-step=479000.ckpt"
]]
# 1gd1l289 u198oqq7 4oiu7bkj 8j9skqvk lyv2aecn
AE_PONG_CKPTS = [jn(CITRIS_CKPT_ROOT, specific_path) for specific_path in [
    "u198oqq7_Autoencoder_pong/epoch=999-step=488000.ckpt",
    "4oiu7bkj_Autoencoder_pong/epoch=999-step=488000.ckpt",
    "1gd1l289_Autoencoder/epoch=979-step=478240.ckpt",
    "8j9skqvk_Autoencoder_pong/epoch=999-step=488000.ckpt",
    "lyv2aecn_Autoencoder_pong/epoch=999-step=488000.ckpt"
]]
CE_CKPT = jn(CITRIS_CKPT_ROOT,"CausalEncoder/p2_citris/1gjsknnr/checkpoints/epoch=199-step=390599.ckpt")
CUDA_DEVICE = 'cuda'
