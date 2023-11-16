import numpy as np
from os.path import join as jn
# Carve out and store a validation set from the training set
from constants import DATASET_ROOT_FOLDER, TRIS_ROOT, TC3DI_ROOT

NUM_VAL = 5000
arr = np.load(jn(TC3DI_ROOT,"og_train.npz"))

# Logic for interventions:
# og_samples:       1 2 3 4
# og_pairs:         1 2 3
#                   2 3 4
# og_intvs:         1 2 3

# split1 samples:   1 2 3
# split1 pairs:     1 2
#                   2 3
# split1 intvs:     1 2

# split2 samples:   3 4
# split2 pairs:     3
#                   4
# split2 intvs:     3

# So here, sample 3 is used once as ct (in split 1), and once as ct+1 (in split 2)

interventions, latents, shape_latents, raw_latents, imgs = arr['interventions'], arr['latents'], arr['shape_latents'], arr['raw_latents'], arr['imgs']
train_interventions, train_latents, train_shape_latents, train_raw_latents, train_imgs = interventions[:-NUM_VAL], latents[:-NUM_VAL], shape_latents[:-NUM_VAL], raw_latents[:-NUM_VAL], imgs[:-NUM_VAL]
val_interventions, val_latents, val_shape_latents, val_raw_latents, val_imgs = interventions[-NUM_VAL:], latents[-NUM_VAL - 1:], shape_latents[-NUM_VAL - 1:], raw_latents[-NUM_VAL - 1:], imgs[-NUM_VAL -1:]
np.savez_compressed(jn(TC3DI_ROOT,"val.npz"), interventions=val_interventions, latents=val_latents, shape_latents=val_shape_latents, raw_latents=val_raw_latents, imgs=val_imgs)
np.savez_compressed(jn(TC3DI_ROOT,"train.npz"), interventions=train_interventions, latents=train_latents, shape_latents=train_shape_latents, raw_latents=train_raw_latents, imgs=train_imgs)