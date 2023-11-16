"""
PyTorch dataset classes for loading the datasets.
"""

import torch
import torch.utils.data as data
import torch.nn.functional as F
import os
import json
import numpy as np
from collections import OrderedDict
from tqdm.auto import tqdm


class MyDataset(data.Dataset):

    def __init__(self, data_folder, split='train', single_image=False, return_latents=False, triplet=False, seq_len=2, coarse_vars=False,mini=False, include_og_imgs=False, cma=False, require_imgs=True, deconf=False,
                 subtest=False, **kwargs):
        super().__init__()
        filename = split
        if triplet:
            if subtest:
                filename += '_subtest'
            filename += '_triplets'
            if coarse_vars:
                filename += '_coarse'
        assert not (deconf and cma), "cma will get a deconfounded dataset WITHOUT interventions, intended for CMA measuring. deconf will get a deconfounded dataset WITH interventions, intended for training."
        if cma:
            filename += '_cma'
        if deconf:
            filename += '_deconf'
        self.split_name = filename
        self.mini = mini
        self.require_imgs = require_imgs
        self.cma = cma
        self.deconf = deconf
        self.iid_pairs = self.cma or self.deconf
        self.maybe_mini = '_mini' if mini else ''
        self.include_og_imgs = include_og_imgs
        self.data_file = os.path.join(data_folder, f'{self.split_name}{self.maybe_mini}.npz')
        if split.startswith('val') and not os.path.isfile(self.data_file):
            self.split_name = self.split_name.replace('val', 'test')
            print(f'[!] WARNING: Could not find a validation dataset for {self.split_name}{self.maybe_mini}.npz. Falling back to the standard test set {self.split_name.replace("val", "test")}{self.maybe_mini}.npz. '
                  f'Do not use it for selecting the best model!')
            self.data_file = os.path.join(data_folder, f'{self.split_name.replace("val", "test")}{self.maybe_mini}.npz')
        assert os.path.isfile(self.data_file), f'Could not find {self.__class__.__name__} dataset at {self.data_file}'



    def load_encodings(self, filename):
        if self.include_og_imgs:
            self.og_imgs = self.imgs
        self.imgs = torch.load(filename)
        self.encodings_active = True


    def maybe_include_og_imgs(self, idx, result, iid_pairs=False):
        if self.encodings_active and self.include_og_imgs:
            og_imgs = self.og_imgs[idx:idx + self.seq_len] if not iid_pairs else self.og_imgs[idx] # if cma, it is already a pair of images (only supports seq_len=2)
            og_imgs = self._prepare_imgs(og_imgs, og=True)
            # result.append(og_imgs)
            if 'imgs' in result:
                result['encs'] = result['imgs']
            result['imgs'] = og_imgs
        return result


    def _prepare_imgs(self, imgs, og=False):
        if imgs == []:
            return []
        if (not og) and self.encodings_active:
            return imgs
        else:
            imgs = imgs.float() / 255.0
            imgs = imgs * 2.0 - 1.0
            return imgs


    @torch.no_grad()
    def encode_dataset(self, encoder, batch_size=256):
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        encoder.eval()
        encoder.to(device)
        encodings = None
        for idx in tqdm(range(0, self.imgs.shape[0], batch_size), desc='Encoding dataset...', leave=False):
            batch = self.imgs[idx:idx+batch_size].to(device)
            batch = self._prepare_imgs(batch)
            if len(batch.shape) == 5:
                batch = batch.flatten(0, 1)
            batch = encoder(batch)
            if len(self.imgs.shape) == 5:
                batch = batch.unflatten(0, (-1, self.imgs.shape[1]))
            batch = batch.detach().cpu()
            if encodings is None:
                encodings = torch.zeros(self.imgs.shape[:-3] + batch.shape[-1:], dtype=batch.dtype, device='cpu')
            encodings[idx:idx+batch_size] = batch
        if self.include_og_imgs:
            self.og_imgs = self.imgs
        self.imgs = encodings
        self.encodings_active = True
        return encodings

    @classmethod
    def get_coarse2fine_correlation_summary(cls, corr_matrix):
        f2c = cls.FINE2COARSE
        fv = cls.FACTORS
        iv = cls.INTERVENED_FACTORS
        col_diag_idxs = list(range(len(fv)))
        row_diag_idxs = []
        for i, true_causal_var in enumerate(fv):
            coarse_var = f2c[true_causal_var]
            if coarse_var in iv:
                row_diag_idxs.append(iv.index(coarse_var))
            else:
                row_diag_idxs.append(len(iv))
        # assert corr_matrix.shape == (len(set(row_diag_idxs)), len(set(col_diag_idxs))), f'Expected shape {(len(row_diag_idxs), len(col_diag_idxs))}, got {corr_matrix.shape}'
        avg_diag = np.mean([corr_matrix[i,j] for i,j in zip(row_diag_idxs, col_diag_idxs)])
        m = corr_matrix.copy()
        m[row_diag_idxs, col_diag_idxs] = -100
        max_off_diag = m.max(axis=0).mean()
        return avg_diag, max_off_diag


class InterventionalPongDataset(MyDataset):

    VAR_INFO = OrderedDict({
        'background': 'categ_2',
        'ball-vel-dir': 'angle',
        'ball-vel-magn': 'continuous_1',
        'ball-x': 'continuous_1',
        'ball-y': 'continuous_1',
        'paddle-left-y': 'continuous_1',
        'paddle-right-y': 'continuous_1',
        'score-left': 'categ_5',
        'score-right': 'categ_5'
    })

    MY_VAR_INFO = OrderedDict({
        k.replace('-', '_'): v for k, v in VAR_INFO.items()
    })

    # region stuff for MFP models and DA

    # Made using following code:
    # import torch
    # torch.manual_seed(5)
    # # assuming 7-dimensional latent factors (excluding ball_vel_magn)
    # mixing_matrix = torch.randn(7, 7)
    # # Make it orthogonal so that it is easily invertible. Based on https://discuss.pytorch.org/t/how-to-efficiently-and-randomly-produce-an-orthonormal-matrix/153609/2
    # mixing_matrix = torch.linalg.qr(mixing_matrix)[0]
    # torch.set_printoptions(precision=25, linewidth=2000);
    # print("MIXING_MATRIX = \\")
    # print(f'torch.{str(mixing_matrix)}')

    MIXING_MATRIX = \
        torch.tensor([  [-0.4883061647415161132812500, -0.3157438635826110839843750,  0.2936354279518127441406250, -0.3678777515888214111328125, -0.4016747176647186279296875,  0.4695419669151306152343750,  0.2418572604656219482421875],
                        [ 0.3809839487075805664062500, -0.4659574329853057861328125, -0.4757396578788757324218750, -0.4024094641208648681640625, -0.1300449520349502563476562, -0.2564941942691802978515625,  0.4083776772022247314453125],
                        [ 0.1550256013870239257812500,  0.6531747579574584960937500, -0.4589737653732299804687500, -0.2042934894561767578125000, -0.2156951725482940673828125,  0.4895974993705749511718750,  0.1034742593765258789062500],
                        [ 0.1594969034194946289062500, -0.1784230917692184448242188, -0.1117847785353660583496094, -0.0826525390148162841796875, -0.5714971423149108886718750, -0.0404804870486259460449219, -0.7714607119560241699218750],
                        [-0.3930485546588897705078125, -0.3225266337394714355468750, -0.6137233376502990722656250,  0.5579925179481506347656250,  0.0313766188919544219970703,  0.2287575751543045043945312, -0.0127681661397218704223633],
                        [-0.2888518571853637695312500, -0.0668819323182106018066406, -0.2322352528572082519531250, -0.5716022849082946777343750,  0.6041809916496276855468750,  0.0753707811236381530761719, -0.4008912146091461181640625],
                        [-0.5735970139503479003906250,  0.3409115672111511230468750, -0.1835966706275939941406250, -0.1268101036548614501953125, -0.2873906493186950683593750, -0.6437281966209411621093750,  0.0894307121634483337402344]])

    FACTORS = [k for k in MY_VAR_INFO if k not in ('background', 'ball_vel_magn')]
    ALT_WRITTEN_FACTORS = ['ball-vel-dir', 'ball-x', 'ball-y', 'paddle-left-y', 'paddle-right-y',
                           'score-left', 'score-right']
    FACTORS_FOR_NO_INTV_CASE = FACTORS + ['ball_vel_magn']
    COARSE_FACTORS = FACTORS
    INTERVENED_FACTORS = [k for k in FACTORS if not 'score' in k]
    INTERVENED_COARSE_FACTORS = INTERVENED_FACTORS
    F2SHORT = {
        'ball_vel_dir': 'bvd',
        'ball_x': 'bx',
        'ball_y': 'by',
        'paddle_left_y': 'ply',
        'paddle_right_y': 'pry',
        'score_left': 'sl',
        'score_right': 'sr',
        'trash': 't'
    }
    SHORT2F = {v: k for k, v in F2SHORT.items()}
    OOD_VARIANTS = FACTORS
    globals().update(
        locals())  # For some reason list comprehension gives NameError: name 'F2SHORT' is not defined if I don't do this
    SHORT_OOD_VARIANTS = [F2SHORT[f] for f in OOD_VARIANTS]
    SHORT_FACTORS = [F2SHORT[f] for f in FACTORS]
    SHORT_COARSE_FACTORS = [F2SHORT[f] for f in COARSE_FACTORS]
    SHORT_INTERVENED_FACTORS = [F2SHORT[f] for f in INTERVENED_FACTORS]
    SHORT_COARSE_FACTORS_AND_TRASH = SHORT_COARSE_FACTORS + ['t']
    SHORT_INTERVENED_FACTORS_AND_TRASH = SHORT_INTERVENED_FACTORS + ['t']
    FINE2COARSE = {f:f for f in FACTORS}
    SHORT_FINE2COARSE = {F2SHORT[k]: F2SHORT[v] for k, v in FINE2COARSE.items()}
    FINE2INTERVENED_AND_TRASH = {f:f for f in FACTORS if f not in ('score_left', 'score_right')} | {'score_left': 'trash', 'score_right': 'trash'}
    SHORT_FINE2INTERVENED_AND_TRASH = {F2SHORT[k]: F2SHORT[v] for k, v in FINE2INTERVENED_AND_TRASH.items()}

    DEFAULT_OOD_FACTORS_LIST = [[F2SHORT[f]] for f in FACTORS]
    DEFAULT_UF_FACTORS_LIST = ([[F2SHORT[f]] for f in FACTORS] +  # for TLP
                           [[f] for f in SHORT_COARSE_FACTORS_AND_TRASH] +  # for full models
                           [[i] for i in range(len(FACTORS))]  # for MTLP
                           )
    DEFAULT_LOSS_FACTORS = SHORT_FACTORS

    DEFAULT_NS_SHOTS = [1, 10, 100, 1000, 10000]
    SHORT_NAME = 'pong'
    # endregion

    def __init__(self, data_folder, split='train', single_image=False, return_latents=False, triplet=False, seq_len=2, causal_vars=None, **kwargs):
        super().__init__(data_folder=data_folder, split=split, single_image=single_image, return_latents=return_latents, triplet=triplet, seq_len=seq_len, causal_vars=causal_vars, **kwargs)
        # filename = split
        # if triplet:
        #     filename += '_triplets'
        # self.split_name = filename
        # data_file = os.path.join(data_folder, f'{self.split_name}.npz')
        # if split.startswith('val') and not os.path.isfile(self.data_file):
        #     self.split_name = self.split_name.replace('val', 'test')
        #     print('[!] WARNING: Could not find a validation dataset. Falling back to the standard test set. Do not use it for selecting the best model!')
        #     data_file = os.path.join(data_folder, f'{self.split_name.replace("val", "test")}.npz')
        # assert os.path.isfile(data_file), f'Could not find ComplexInterventionalPong dataset at {data_file}'

        arr = np.load(self.data_file)
        self.imgs = torch.from_numpy(arr['images'])
        self.latents = torch.from_numpy(arr['latents'])
        self.targets = torch.from_numpy(arr['targets'])
        self.keys = [key.replace('_', '-') for key in arr['keys'].tolist()]
        self._clean_up_data(causal_vars)

        self.single_image = single_image
        self.return_latents = return_latents
        self.triplet = triplet
        self.encodings_active = False
        self.seq_len = seq_len if not (single_image or triplet) else 1

    def _clean_up_data(self, causal_vars=None):
        if len(self.imgs.shape) == 5:
            self.imgs = self.imgs.permute(0, 1, 4, 2, 3)  # Push channels to PyTorch dimension
        else:
            self.imgs = self.imgs.permute(0, 3, 1, 2)

        all_latents, all_targets = [], []
        target_names = [] if causal_vars is None else causal_vars
        keys_var_info = list(InterventionalPongDataset.VAR_INFO.keys())
        for key in keys_var_info:
            if key not in self.keys:
                InterventionalPongDataset.VAR_INFO.pop(key)
        for i, key in enumerate(self.keys):
            if key.endswith('-proj'):
                continue
            latent = self.latents[...,i]
            target = self.targets[...,i]
            if key == 'ball-vel-magn' and latent.unique().shape[0] == 1:
                if key in InterventionalPongDataset.VAR_INFO:
                    InterventionalPongDataset.VAR_INFO.pop(key)
                continue
            if InterventionalPongDataset.VAR_INFO[key].startswith('continuous'):
                if key.endswith('-x') or key.endswith('-y'):
                    latent = latent / 16.0 - 1.0
                else:
                    latent = latent - 2.0
            if causal_vars is not None:
                if key in causal_vars:
                    all_targets.append(target)
            elif target.sum() > 0:
                all_targets.append(target)
                target_names.append(key)
            all_latents.append(latent)
        self.latents = torch.stack(all_latents, dim=-1)
        # self.targets = torch.stack(all_targets, dim=-1)
        if causal_vars is None: # Nathan: for never-intervened data whose alignment I still want to check
            if self.targets.sum() == 0:
                self.targets = self.targets
            else:
                self.targets = torch.stack(all_targets, dim=-1)
        else:
            if len(causal_vars) == 0:
                self.targets = self.targets
            else:
                self.targets = torch.stack(all_targets, dim=-1)
        self.target_names_l = target_names
        print(f'Using the causal variables {self.target_names_l}')
        self.full_target_names = self.target_names_l[:]

    @torch.no_grad()
    def encode_dataset(self, encoder, batch_size=512):
        return super().encode_dataset(encoder, batch_size)
        # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        # encoder.eval()
        # encoder.to(device)
        # encodings = None
        # for idx in tqdm(range(0, self.imgs.shape[0], batch_size), desc='Encoding dataset...', leave=False):
        #     batch = self.imgs[idx:idx+batch_size].to(device)
        #     batch = self._prepare_imgs(batch)
        #     if len(batch.shape) == 5:
        #         batch = batch.flatten(0, 1)
        #     batch = encoder(batch)
        #     if len(self.imgs.shape) == 5:
        #         batch = batch.unflatten(0, (-1, self.imgs.shape[1]))
        #     batch = batch.detach().cpu()
        #     if encodings is None:
        #         encodings = torch.zeros(self.imgs.shape[:-3] + batch.shape[-1:], dtype=batch.dtype, device='cpu')
        #     encodings[idx:idx+batch_size] = batch
        # self.imgs = encodings
        # self.encodings_active = True
        # return encodings

    # def load_encodings(self, filename):
    #     self.imgs = torch.load(filename)
    #     self.encodings_active = True

    # def _prepare_imgs(self, imgs):
    #     if self.encodings_active:
    #         return imgs
    #     else:
    #         imgs = imgs.float() / 255.0
    #         imgs = imgs * 2.0 - 1.0
    #         return imgs

    def label_to_img(self, label):
        return (label + 1.0) / 2.0

    def num_labels(self):
        return -1

    def num_vars(self):
        return self.targets.shape[-1]

    def target_names(self):
        return self.target_names_l

    def get_img_width(self):
        return self.imgs.shape[-2]

    def get_inp_channels(self):
        return self.imgs.shape[-3]

    def get_causal_var_info(self):
        return InterventionalPongDataset.VAR_INFO

    def __len__(self):
        return self.imgs.shape[0] - self.seq_len + 1

    def __getitem__(self, idx):
        # returns = []
        result = {}
        if self.triplet:
            img_or_imgs = self.imgs[idx]
            pos = self.latents[idx]
            target = self.targets[idx]
        else:
            img_or_imgs = self.imgs[idx:idx+self.seq_len]
            pos = self.latents[idx:idx+self.seq_len]
            target = self.targets[idx:idx+self.seq_len-1]

        if self.single_image:
            img_or_imgs = img_or_imgs[0]
            pos = pos[0]
        else:
            # returns += [target]
            result['targets'] = target
        img_or_imgs = self._prepare_imgs(img_or_imgs)
        # returns = [img_pair] + returns
        img_key = 'img' if self.single_image else ('imgs' if not self.encodings_active else 'encs')
        result[img_key] = img_or_imgs

        if self.return_latents:
            # returns += [pos]
            result['lat'] = pos

        result = self.maybe_include_og_imgs(idx, result)

        # return tuple(returns) if len(returns) > 1 else returns[0]
        result['isTriplet'] = self.triplet
        return result


class Causal3DDataset(MyDataset):

    VAR_INFO = OrderedDict({
        'pos-x': 'continuous_2',
        'pos-y': 'continuous_2',
        'pos-z': 'continuous_2',
        'rot-alpha': 'angle',
        'rot-beta': 'angle',
        'rot-gamma': 'angle',
        'rot-spot': 'angle',
        'hue-object': 'categ_8',
        'hue-spot': 'categ_8',
        'hue-back': 'categ_8',
        'obj-shape': 'categ_7',
        'obj-material': 'categ_3'
    })


    MY_VAR_INFO = OrderedDict({
        'x_pos': 'continuous_2',
        'y_pos': 'continuous_2',
        'z_pos': 'continuous_2',
        'alpha': 'angle',
        'beta': 'angle',
'gamma': 'angle',
        'rot_spotlight': 'angle',
        'hue_object': 'angle',
        'hue_spotlight': 'angle',
        'hue_background': 'angle',
        'shape': 'categ_7',
        'material': 'categ_3'
    })


    # region stuff for MFP models and DA

    # Made using following code:
    # torch.manual_seed(5)
    # # assuming 10-dimensional latent factors (excluding gamma and obj-material)
    # mixing_matrix = torch.randn(10, 10)
    # # Make it orthogonal so that it is easily invertible. Based on https://discuss.pytorch.org/t/how-to-efficiently-and-randomly-produce-an-orthonormal-matrix/153609/2
    # mixing_matrix = torch.linalg.qr(mixing_matrix)[0]
    # torch.set_printoptions(precision=25, linewidth=2000);
    # print(mixing_matrix);
    # print(unmixing_matrix)
    # print("MIXING_MATRIX = \\")
    # print(f'torch.{str(mixing_matrix)}')

    MIXING_MATRIX = torch.tensor(
        [[-6.5995562076568603515625000e-01, -1.9643338024616241455078125e-01,  3.4676072001457214355468750e-01, -5.0139480829238891601562500e-01,  1.2699176371097564697265625e-01, -2.6220321655273437500000000e-01, -1.5349552035331726074218750e-01,  2.0684532821178436279296875e-02, -5.6169133633375167846679688e-02,  2.0545460283756256103515625e-01],
        [ 4.8455694317817687988281250e-01, -1.2655678438022732734680176e-04, -1.5257117152214050292968750e-01, -1.4404666423797607421875000e-01,  3.3671411871910095214843750e-01, -5.0331032276153564453125000e-01, -5.2304124832153320312500000e-01,  2.0259836316108703613281250e-01, -9.8074793815612792968750000e-02,  1.7390456795692443847656250e-01],
        [ 4.6447504311800003051757812e-02, -2.1941258013248443603515625e-01, -8.7911210954189300537109375e-02, -1.1966251581907272338867188e-01, -6.2371537089347839355468750e-02,  5.3651207685470581054687500e-01, -9.1810308396816253662109375e-02,  4.3338561058044433593750000e-01, -6.1905890703201293945312500e-01,  2.3755353689193725585937500e-01],
        [-8.9274644851684570312500000e-02,  6.6534167528152465820312500e-01,  2.1410115063190460205078125e-01, -4.6715792268514633178710938e-02,  5.5565607547760009765625000e-01,  1.8388712406158447265625000e-01,  7.1840500459074974060058594e-03,  2.9072185978293418884277344e-02, -2.5737196207046508789062500e-01, -3.0269402265548706054687500e-01],
        [ 5.2978146821260452270507812e-02,  4.0296316146850585937500000e-01, -1.4690683782100677490234375e-01, -5.5316489934921264648437500e-01, -4.3797686696052551269531250e-01,  1.9559991359710693359375000e-01, -2.3777933418750762939453125e-01,  2.5923106074333190917968750e-01,  3.6988902091979980468750000e-01, -1.2884265184402465820312500e-01],
        [-7.7459163963794708251953125e-02,  4.0150213241577148437500000e-01, -5.1871906965970993041992188e-02,  6.9955743849277496337890625e-02, -5.0466620922088623046875000e-01, -4.9604162573814392089843750e-01,  2.0653431117534637451171875e-01,  1.1616635136306285858154297e-02, -5.2435964345932006835937500e-01,  8.1989511847496032714843750e-02],
        [-3.5486656427383422851562500e-01,  8.5660964250564575195312500e-02, -8.6096078157424926757812500e-01, -7.5169719755649566650390625e-02,  2.6505503058433532714843750e-01, -9.7196921706199645996093750e-03,  1.3605217635631561279296875e-01, -8.5864663124084472656250000e-02,  2.2689070552587509155273438e-02,  1.5192565321922302246093750e-01],
        [ 3.1448888778686523437500000e-01, -1.1077996343374252319335938e-01,  5.4678201675415039062500000e-02, -4.2788186669349670410156250e-01,  1.9156524538993835449218750e-01, -1.4898665249347686767578125e-01,  7.5446152687072753906250000e-01,  2.6415544748306274414062500e-01,  6.8149417638778686523437500e-02,  1.4934691600501537322998047e-02],
        [ 2.9217883944511413574218750e-01,  2.3361954838037490844726562e-02,  9.4605069607496261596679688e-03, -4.1619750857353210449218750e-01, -3.4948643296957015991210938e-02,  1.8519958853721618652343750e-01, -1.8090220168232917785644531e-02, -7.8749829530715942382812500e-01, -2.1762278676033020019531250e-01,  1.9343008100986480712890625e-01],
        [ 2.2957000881433486938476562e-02,  3.5616514086723327636718750e-01,  1.8521894514560699462890625e-01,  1.9624862074851989746093750e-01,  5.6288480758666992187500000e-02,  1.2394288927316665649414062e-01,  8.4505386650562286376953125e-02,  7.2234071791172027587890625e-02,  2.7104017138481140136718750e-01,  8.3393526077270507812500000e-01]])

    FACTORS = ['x_pos', 'y_pos', 'z_pos', 'alpha', 'beta', 'rot_spotlight', 'hue_object', 'hue_spotlight',
               'hue_background', 'shape']\

    # region ID_PARENTAGE and OOD_PARENTAGE
    # dict where keys are children and values list of parents
    ID_PARENTAGE = {
        'x_pos': ['x_pos','beta'],
        'y_pos': ['y_pos','alpha'],
        'z_pos': ['z_pos','alpha'],
        'alpha': ['alpha', 'hue_background'],
        'beta': ['beta', 'hue_object'],
        'rot_spotlight': ['rot_spotlight','x_pos', 'y_pos'],
        'hue_object': ['hue_object','hue_spotlight','hue_background', 'shape'],
        'hue_spotlight': ['hue_spotlight','hue_background'],
        'hue_background': ['hue_background'],
        'shape': ['shape']
    }
    OOD_PARENTAGE = ID_PARENTAGE.copy()
    # x_pos, y_pos, hue_object, hue_spotlight, shape have same parentage
    OOD_PARENTAGE['z_pos'] = ['z_pos','beta']
    OOD_PARENTAGE['alpha'] = ['alpha','hue_object']
    OOD_PARENTAGE['beta'] = ['beta','hue_background']
    OOD_PARENTAGE['rot_spotlight'] = ['rot_spotlight','y_pos', 'z_pos']
    OOD_PARENTAGE['hue_background'] = ['hue_background','hue_object']
    # endregion

    # For testing new-shapes generalization
    OOD_VARIANTS = FACTORS + ['shape_no_35', 'shape_no_01246']
    ALT_WRITTEN_FACTORS = ['pos-x', 'pos-y', 'pos-z', 'rot-alpha', 'rot-beta', 'rot-spot', 'hue-object', 'hue-spot',
                           'hue-back', 'obj-shape'] # This is how they're written in triplet dist logging
    COARSE_FACTORS = ['pos', 'rot', 'rot_spotlight', 'hue_object', 'hue_spotlight', 'hue_background', 'shape']

    INTERVENED_FACTORS = COARSE_FACTORS
    FACTORS_FOR_NO_INTV_CASE = COARSE_FACTORS
    F2SHORT = {
        'pos': 'p',
        'x_pos': 'x',
        'y_pos': 'y',
        'z_pos': 'z',
        'rot': 'r',
        'alpha': 'a',
        'beta': 'b',
        'rot_spotlight': 'rs',
        'hue_object': 'ho',
        'hue_spotlight': 'hs',
        'hue_background': 'hb',
        'shape': 's',
        'trash': 't',
        'shape_no_01246': 'sno01246',
        'shape_no_35': 'sno35',
    }
    SHORT2F = {v: k for k, v in F2SHORT.items()}
    globals().update(locals()) # For some reason list comprehension gives NameError: name 'F2SHORT' is not defined if I don't do this
    SHORT_FACTORS = [F2SHORT[f] for f in FACTORS]
    SHORT_OOD_VARIANTS = [F2SHORT[f] for f in OOD_VARIANTS]
    SHORT_COARSE_FACTORS = [F2SHORT[f] for f in COARSE_FACTORS]
    SHORT_COARSE_FACTORS_AND_TRASH = SHORT_COARSE_FACTORS + ['t']
    SHORT_INTERVENED_FACTORS = [F2SHORT[f] for f in INTERVENED_FACTORS]
    SHORT_INTERVENED_FACTORS_AND_TRASH = SHORT_INTERVENED_FACTORS + ['t']
    FINE2COARSE = {
        'x_pos': 'pos',
        'y_pos': 'pos',
        'z_pos': 'pos',
        'alpha': 'rot',
        'beta': 'rot',
        'rot_spotlight': 'rot_spotlight',
        'hue_object': 'hue_object',
        'hue_spotlight': 'hue_spotlight',
        'hue_background': 'hue_background',
        'shape': 'shape',
    }
    SHORT_FINE2COARSE = {F2SHORT[k]: F2SHORT[v] for k, v in FINE2COARSE.items()}
    DEFAULT_OOD_FACTORS_LIST = [[F2SHORT[f]] for f in FACTORS if f != 'shape']
    DEFAULT_UF_FACTORS_LIST = ([[F2SHORT[f]] for f in FACTORS if f != 'shape'] +  # for TLP
                           [[f] for f in SHORT_COARSE_FACTORS_AND_TRASH if SHORT2F[f] != 'shape'] +  # for full models
                           [[i] for i in range(len(FACTORS))]  # for MTLP
                           )
    DEFAULT_LOSS_FACTORS = SHORT_FACTORS
    SHORT_NAME = 'shapes'
    # DEFAULT_NS_SHOTS = [1, 10, 50, 100, 200, 500, 1000]
    # DEFAULT_NS_SHOTS = [1, 10, 50, 100, 200, 500, 1000, 10000, 100000]
    # DEFAULT_NS_SHOTS = [1, 50, 200, 1000, 10000, 100000]
    DEFAULT_NS_SHOTS = [1, 20, 50, 100, 200, 1000, 10000, 100000]
    #endregion

    def __init__(self, data_folder, split='train', single_image=False, seq_len=2, coarse_vars=False, triplet=False, causal_vars=None, return_latents=False, img_width=-1, exclude_vars=None, max_dataset_size=-1, exclude_objects=None,mini=False, also_intervened_latents=False, cma=False, **kwargs):
        assert (not also_intervened_latents) or return_latents, 'Cannot give intervened latents if not returning latents'
        super().__init__(data_folder, split, single_image, seq_len=seq_len, coarse_vars=coarse_vars, triplet=triplet, causal_vars=causal_vars, return_latents=return_latents, img_width=img_width, exclude_vars=exclude_vars, max_dataset_size=max_dataset_size, exclude_objects=exclude_objects, mini=mini, also_intervened_latents=also_intervened_latents, cma=cma, **kwargs)
        # filename = split
        # if triplet:
        #     filename += '_triplets'
        #     if coarse_vars:
        #         filename += '_coarse'
        # self.split_name = filename
        # self.mini = mini
        # self.maybe_mini = '_mini' if mini else ''
        # data_file = os.path.join(data_folder, f'{self.split_name}{self.maybe_mini}.npz')
        # if split.startswith('val') and not os.path.isfile(self.data_file):
        #     self.split_name = self.split_name.replace('val', 'test')
        #     print(f'[!] WARNING: Could not find a validation dataset for {self.split_name}{self.maybe_mini}.npz. Falling back to the standard test set {self.split_name.replace("val", "test")}{self.maybe_mini}.npz. '
        #           f'Do not use it for selecting the best model!')
        #     data_file = os.path.join(data_folder, f'{self.split_name.replace("val", "test")}{self.maybe_mini}.npz')
        # assert os.path.isfile(data_file), f'Could not find causal3d dataset at {data_file}'

        self.triplet = triplet
        arr = np.load(self.data_file)
        if self.require_imgs:
            self.imgs = self.load_imgs(arr, img_width)
        self.train = (split == 'train')
        self.single_image = single_image
        self.coarse_vars = coarse_vars
        self.seq_len = seq_len if not (single_image or triplet) else 1
        self.return_latents = return_latents
        self.max_dataset_size = max_dataset_size
        self.encodings_active = False
        self._prepare_causal_vars(arr, coarse_vars, causal_vars, exclude_vars,also_intervened_latents)
        if self.require_imgs:
            self.sub_indices = torch.arange(self.imgs.shape[0] - (0 if self.iid_pairs else (self.seq_len - 1)), dtype=torch.long)
        else:
            self.sub_indices = torch.arange(self.true_latents.shape[0] - (0 if self.iid_pairs else (self.seq_len - 1)), dtype=torch.long)
        self.obj_triplet_indices = None
        if exclude_objects is not None:
            imgs_to_remove = torch.stack([self.true_latents[...,-1] == o for o in exclude_objects], dim=0).any(dim=0)
            if self.triplet:
                same_obj = (self.true_latents[:,:1,-1] == self.true_latents[:,:,-1]).all(dim=1)
                imgs_to_remove = torch.logical_and(~same_obj, imgs_to_remove.any(dim=1))
                self.obj_triplet_indices = (self.true_latents[:,-1:,-1] == torch.FloatTensor(exclude_objects)[None,:])
                self.obj_triplet_indices = (self.obj_triplet_indices.long() * torch.arange(1, 3, device=self.true_latents.device, dtype=torch.long)[None]).max(dim=-1).values
                
            for i in range(1, self.seq_len):
                imgs_to_remove[:-i] = torch.logical_or(imgs_to_remove[i:], imgs_to_remove[:-i])
            self.sub_indices = self.sub_indices[~imgs_to_remove[:self.sub_indices.shape[0]]]
            og_size = self.imgs.shape[0] if self.require_imgs else self.true_latents.shape[0]
            print(f'Removed images from {self.split_name} [objs {exclude_objects}]: {og_size} -> {self.sub_indices.shape[0]}')

    def load_imgs(self, arr, img_width):
        if not self.iid_pairs:
            imgs = torch.from_numpy(arr['imgs'])[..., :3]
        else:
            try:
                imgs = torch.from_numpy(np.stack([arr['in_imgs'], arr['out_imgs']], axis=1))[..., :3]
            except ValueError as e:
                print("Could not find in_imgs and out_imgs in dataset. Likely they haven't been generated yet. ")
                raise e
        if not (self.triplet or self.iid_pairs):
            # Shape: (N, H, W, C) -> (N, C, H, W)
            imgs = imgs.permute(0, 3, 1, 2)
        else:
            # Shape: (N, 2, H, W, C) -> (N, 2, C, H, W)
            imgs = imgs.permute(0, 1, 4, 2, 3)
        if img_width > 0 and img_width != imgs.shape[-1]:
            full_shape = imgs.shape
            dtype = imgs.dtype
            if len(imgs.shape) == 5:
                imgs = imgs.flatten(0, 1)
            imgs = F.interpolate(imgs.float(), size=(img_width, img_width), mode='bilinear')
            imgs = imgs.reshape(full_shape[:-2] + (img_width, img_width))
            imgs = imgs.to(dtype)
        return imgs

    def _prepare_causal_vars(self, arr, coarse_vars=False, causal_vars=None, exclude_vars=None, also_intervened_latents=False):
        target_names_l = list(Causal3DDataset.VAR_INFO.keys()) # pos-x, pos-y, pos-z, rot-alpha, rot-beta, rot-gamma, rot-spot, hue-object, hue-spot, hue-back, obj-shape, obj-material
        targets = torch.from_numpy(arr['interventions'])
        if not self.iid_pairs:
            true_latents = torch.cat([torch.from_numpy(arr['raw_latents']), torch.from_numpy(arr['shape_latents'])], dim=-1)
        else:
            true_latents = torch.stack([torch.cat([torch.from_numpy(arr['raw_in_latents']), torch.from_numpy(arr['in_shape_latents'])], dim=-1),
                                        torch.cat([torch.from_numpy(arr['raw_out_latents']), torch.from_numpy(arr['out_shape_latents'])], dim=-1)], dim=1)
        assert targets.shape[-1] == len(target_names_l), f'We have {len(target_names_l)} target names, but the intervention vector has only {targets.shape[-1]} entries.'
        if not causal_vars:
            has_intv = torch.logical_and((targets == 0).any(dim=0), (targets == 1).any(dim=0))
            has_intv = torch.logical_and(has_intv, (true_latents[0:1] != true_latents).any(dim=0))
            if self.cma:
                has_intv = has_intv[0]
        else:
            has_intv = torch.Tensor([(n in causal_vars) for n in target_names_l]).bool()
        
        self.true_latents = true_latents[...,has_intv] if not also_intervened_latents else true_latents
        self.target_names_l = [n for i, n in enumerate(target_names_l) if has_intv[i]]
        self.full_target_names = self.target_names_l[:]
        for i, name in enumerate(self.target_names_l):
            if name.startswith('hue') and Causal3DDataset.VAR_INFO[name].startswith('categ'):
                orig_shape = self.true_latents.shape
                self.true_latents = self.true_latents.flatten(0, -2)
                unique_vals, _ = torch.unique(self.true_latents[:,i]).sort()
                if unique_vals.shape[0] > 50:  # Replace categorical by angle
                    # print(f'-> Changing {name} to angles')
                    Causal3DDataset.VAR_INFO[name] = 'angle'
                else:
                    num_categs = (unique_vals[None] == self.true_latents[:,i:i+1]).float().sum(dim=0)
                    unique_vals = unique_vals[num_categs > 10]
                    assert unique_vals.shape[0] <= int(Causal3DDataset.VAR_INFO[name].split('_')[-1])
                    self.true_latents[:,i] = (unique_vals[None] == self.true_latents[:,i:i+1]).float().argmax(dim=-1).float()
                self.true_latents = self.true_latents.reshape(orig_shape)

        if exclude_vars is not None:
            for v in exclude_vars:
                assert v in target_names_l, f'Could not find \"{v}\" in the name list: {target_names_l}.'
                idx = target_names_l.index(v)
                if self.triplet and (targets[:,idx] == 1).any():
                    print(f'[!] WARNING: Triplets will not work properly for \"{v}\"')
                has_intv[idx] = False

        self.targets = targets[...,has_intv]
        self.target_names_l = [n for i, n in enumerate(target_names_l) if has_intv[i]] # This line seems redundant/double.
        if coarse_vars:
            for abs_class in ['rot', 'pos']:
                abs_vars = torch.Tensor([n.startswith(f'{abs_class}-') and not n.endswith('spot') for n in self.target_names_l]).bool()
                self.targets = torch.cat([self.targets[...,abs_vars].any(dim=-1, keepdims=True), self.targets[...,~abs_vars]], dim=-1)
                self.target_names_l = [abs_class] + [n for i, n in enumerate(self.target_names_l) if not abs_vars[i]]
        # Nathan
        default_target_names_l = ['pos', 'rot', 'rot-spot', 'hue-object', 'hue-spot', 'hue-back', 'obj-shape']
        if self.target_names_l != default_target_names_l:
            print(f'Considering the following causal variables: {self.target_names_l}')
    #
    # @torch.no_grad()
    # def encode_dataset(self, encoder, batch_size=256):
    #     device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    #     encoder.eval()
    #     encoder.to(device)
    #     encodings = None
    #     for idx in tqdm(range(0, self.imgs.shape[0], batch_size), desc='Encoding dataset...', leave=False):
    #         batch = self.imgs[idx:idx+batch_size].to(device)
    #         batch = self._prepare_imgs(batch)
    #         if len(batch.shape) == 5:
    #             batch = batch.flatten(0, 1)
    #         batch = encoder(batch)
    #         if len(self.imgs.shape) == 5:
    #             batch = batch.unflatten(0, (-1, self.imgs.shape[1]))
    #         batch = batch.detach().cpu()
    #         if encodings is None:
    #             encodings = torch.zeros(self.imgs.shape[:-3] + batch.shape[-1:], dtype=batch.dtype, device='cpu')
    #         encodings[idx:idx+batch_size] = batch
    #     if self.include_og_imgs:
    #         self.og_imgs = self.imgs
    #     self.imgs = encodings
    #     self.encodings_active = True
    #     return encodings

    def num_labels(self):
        return -1

    def num_vars(self):
        return self.targets.shape[-1]

    def target_names(self):
        return self.target_names_l

    def get_img_width(self):
        return self.imgs.shape[-1]

    def get_inp_channels(self):
        return self.imgs.shape[-3]

    def get_causal_var_info(self):
        causal_var_info = OrderedDict()
        for key in self.target_names_l:
            causal_var_info[key] = Causal3DDataset.VAR_INFO[key]
        return causal_var_info

    def label_to_img(self, label):
        return (label + 1.0) / 2.0

    def __len__(self):
        if self.max_dataset_size > 0:
            return min(self.max_dataset_size, self.sub_indices.shape[0])
        else:
            return self.sub_indices.shape[0]

    def __getitem__(self, idx):
        idx = self.sub_indices[idx]
        result = {}
        if self.single_image:
            if self.require_imgs:
                img = self.imgs[idx]
                img = self._prepare_imgs(img)
            else:
                img = []
            if self.return_latents:
                lat = self.true_latents[idx]
                # return img, lat
                result['img'] = img
                result['lat'] = lat
            else:
                # return img
                result['img'] = img
        elif self.triplet or self.iid_pairs:
            targets = self.targets[idx] if not self.iid_pairs else self.targets[idx:idx+self.seq_len-1] # cma: to be in line with below
            imgs = self.imgs[idx] if self.require_imgs else []
            imgs = self._prepare_imgs(imgs)
            # if self.return_latents:
            #     lat = self.true_latents[idx]
            #     # result = [imgs, targets, lat]
            #     result |= {'imgs': imgs, 'target': targets, 'lat': lat}
            #     if self.obj_triplet_indices is not None:
            #         # result += [self.obj_triplet_indices[idx]]
            #         result |= {'obj_triplet_indices': self.obj_triplet_indices[idx]}
            # else:
            #     # result = [imgs, targets]
            #     result |= {'imgs': imgs, 'target': targets}
            # if self.iid_pairs:
            #     result = self.maybe_include_og_imgs(idx, result, iid_pairs=True)
            result['targets'] = targets
            result['encs' if self.encodings_active else 'imgs'] = imgs
            if self.return_latents:
                result['lat'] = self.true_latents[idx]
                if self.obj_triplet_indices is not None:
                    result['obj_triplet_indices'] = self.obj_triplet_indices[idx]
            if self.iid_pairs:
                result = self.maybe_include_og_imgs(idx, result, iid_pairs=True)
        else:
            imgs = self.imgs[idx:idx+self.seq_len] if self.require_imgs else []
            targets = self.targets[idx:idx+self.seq_len-1]
            imgs = self._prepare_imgs(imgs)
            # result = [imgs, targets]
            result['encs' if self.encodings_active else 'imgs'] = imgs
            result['targets'] = targets
            if self.return_latents:
                lat = self.true_latents[idx:idx+self.seq_len]
                # result.append(lat)
                result['lat'] = lat

            result = self.maybe_include_og_imgs(idx, result)
        result['isTriplet'] = self.triplet
        return result

class BallInBoxesDataset(data.Dataset):

    VAR_INFO = OrderedDict({
        'ball-b': 'categ_2',
        'ball-x': 'continuous_1',
        'ball-y': 'continuous_1'
    })

    def __init__(self, data_folder, split='train', single_image=False, return_latents=False, triplet=False, seq_len=2, causal_vars=None, **kwargs):
        super().__init__()
        filename = split
        if triplet:
            filename += '_triplets'
        self.split_name = filename
        data_file = os.path.join(data_folder, f'{filename}.npz')
        if split.startswith('val') and not os.path.isfile(data_file):
            self.split_name = self.split_name.replace('val', 'test')
            print('[!] WARNING: Could not find a validation dataset. Falling back to the standard test set. Do not use it for selecting the best model!')
            data_file = os.path.join(data_folder, f'{filename.replace("val", "test")}.npz')
        assert os.path.isfile(data_file), f'Could not find BallInBoxesDataset dataset at {data_file}'

        arr = np.load(data_file)
        self.imgs = torch.from_numpy(arr['images'])
        self.latents = torch.from_numpy(arr['latents'])
        self.targets = torch.from_numpy(arr['targets'])
        self.keys = [key.replace('_', '-') for key in arr['keys'].tolist()]
        self._clean_up_data(causal_vars)

        self.single_image = single_image
        self.return_latents = return_latents
        self.triplet = triplet
        self.encodings_active = False
        self.seq_len = seq_len if not (single_image or triplet) else 1

    def _clean_up_data(self, causal_vars=None):
        if len(self.imgs.shape) == 5:
            self.imgs = self.imgs.permute(0, 1, 4, 2, 3)  # Push channels to PyTorch dimension
        else:
            self.imgs = self.imgs.permute(0, 3, 1, 2)

        all_latents, all_targets = [], []
        target_names = [] if causal_vars is None else causal_vars
        keys_var_info = list(BallInBoxesDataset.VAR_INFO.keys())
        for key in keys_var_info:
            if key not in self.keys:
                BallInBoxesDataset.VAR_INFO.pop(key)
        for i, key in enumerate(self.keys):
            if key.endswith('-proj'):
                continue
            latent = self.latents[...,i]
            target = self.targets[...,i]
            if BallInBoxesDataset.VAR_INFO[key].startswith('continuous'):
                if key.endswith('-y'):
                    latent = (latent / 16.0 - 1.0) / 0.79
                elif key.endswith('-x'):
                    latent = (latent / 8.0 - 1.0) / 0.57
                else:
                    latent = latent - 2.0
            if causal_vars is not None:
                if key in causal_vars:
                    all_targets.append(target)
            elif target.sum() > 0:
                all_targets.append(target)
                target_names.append(key)
            all_latents.append(latent)
        self.latents = torch.stack(all_latents, dim=-1)
        self.targets = torch.stack(all_targets, dim=-1)
        self.target_names_l = target_names
        print(f'Using the causal variables {self.target_names_l}')

    @torch.no_grad()
    def encode_dataset(self, encoder, batch_size=512):
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        encoder.eval()
        encoder.to(device)
        encodings = None
        for idx in tqdm(range(0, self.imgs.shape[0], batch_size), desc='Encoding dataset...', leave=False):
            batch = self.imgs[idx:idx+batch_size].to(device)
            batch = self._prepare_imgs(batch)
            if len(batch.shape) == 5:
                batch = batch.flatten(0, 1)
            batch = encoder(batch)
            if len(self.imgs.shape) == 5:
                batch = batch.unflatten(0, (-1, self.imgs.shape[1]))
            batch = batch.detach().cpu()
            if encodings is None:
                encodings = torch.zeros(self.imgs.shape[:-3] + batch.shape[-1:], dtype=batch.dtype, device='cpu')
            encodings[idx:idx+batch_size] = batch
        self.imgs = encodings
        self.encodings_active = True
        return encodings

    def load_encodings(self, filename):
        self.imgs = torch.load(filename)
        self.encodings_active = True

    def _prepare_imgs(self, imgs):
        if self.encodings_active:
            return imgs
        else:
            imgs = imgs.float() / 255.0
            imgs = imgs * 2.0 - 1.0
            return imgs

    def label_to_img(self, label):
        return (label + 1.0) / 2.0

    def num_labels(self):
        return -1

    def num_vars(self):
        return self.targets.shape[-1]

    def target_names(self):
        return self.target_names_l

    def get_img_width(self):
        return self.imgs.shape[-2]

    def get_inp_channels(self):
        return self.imgs.shape[-3]

    def get_causal_var_info(self):
        return BallInBoxesDataset.VAR_INFO

    def __len__(self):
        return self.imgs.shape[0] - self.seq_len + 1

    def __getitem__(self, idx):
        returns = []

        if self.triplet:
            img_pair = self.imgs[idx]
            pos = self.latents[idx]
            target = self.targets[idx]
        else:
            img_pair = self.imgs[idx:idx+self.seq_len]
            pos = self.latents[idx:idx+self.seq_len]
            target = self.targets[idx:idx+self.seq_len-1]

        if self.single_image:
            img_pair = img_pair[0]
            pos = pos[0]
        else:
            returns += [target]
        img_pair = self._prepare_imgs(img_pair)
        returns = [img_pair] + returns

        if self.return_latents:
            returns += [pos]

        return tuple(returns) if len(returns) > 1 else returns[0]


class VoronoiDataset(data.Dataset):

    VAR_INFO = OrderedDict({
        'c0': 'continuous_2.8',
        'c1': 'continuous_2.8',
        'c2': 'continuous_2.8',
        'c3': 'continuous_2.8',
        'c4': 'continuous_2.8',
        'c5': 'continuous_2.8',
        'c6': 'continuous_2.8',
        'c7': 'continuous_2.8',
        'c8': 'continuous_2.8'
    })

    def __init__(self, data_folder, split='train', single_image=False, return_latents=False, triplet=False, seq_len=2, causal_vars=None, **kwargs):
        super().__init__()
        filename = split
        if triplet:
            filename += '_triplets'
        self.split_name = filename
        data_file = os.path.join(data_folder, f'{filename}.npz')
        if split.startswith('val') and not os.path.isfile(data_file):
            self.split_name = self.split_name.replace('val', 'test')
            print('[!] WARNING: Could not find a validation dataset. Falling back to the standard test set. Do not use it for selecting the best model!')
            data_file = os.path.join(data_folder, f'{filename.replace("val", "test")}.npz')
        assert os.path.isfile(data_file), f'Could not find VoronoiDataset dataset at {data_file}'

        arr = np.load(data_file)
        self.imgs = torch.from_numpy(arr['images'])
        self.latents = torch.from_numpy(arr['latents'])
        self.targets = torch.from_numpy(arr['targets'])
        self.keys = [key.replace('_', '-') for key in arr['keys'].tolist()]
        self._load_settings(data_folder)
        self._clean_up_data(causal_vars)

        self.return_latents = return_latents
        self.triplet = triplet
        self.single_image = single_image
        self.encodings_active = False
        self.seq_len = seq_len if not (single_image or triplet) else 1

    def _clean_up_data(self, causal_vars=None):
        if len(self.imgs.shape) == 5:
            self.imgs = self.imgs.permute(0, 1, 4, 2, 3)  # Push channels to PyTorch dimension
        else:
            self.imgs = self.imgs.permute(0, 3, 1, 2)

        all_latents, all_targets = [], []
        target_names = [] if causal_vars is None else causal_vars
        keys_var_info = list(VoronoiDataset.VAR_INFO.keys())
        for key in keys_var_info:
            if key not in self.keys:
                VoronoiDataset.VAR_INFO.pop(key)
        for i, key in enumerate(self.keys):
            latent = self.latents[...,i]
            if self.settings['graph_idx'] < 0:
                print('Latent std', latent.std().item(), 'Max', latent.max().item(), 'Min', latent.min().item())
                # latent = torch.tanh(latent / 1.5) * 2.5
            target = self.targets[...,i]
            if causal_vars is not None:
                if key in causal_vars:
                    all_targets.append(target)
            elif target.sum() > 0:
                all_targets.append(target)
                target_names.append(key)
            all_latents.append(latent)
        self.latents = torch.stack(all_latents, dim=-1)
        self.targets = torch.stack(all_targets, dim=-1)
        self.target_names_l = target_names
        print(f'Using the causal variables {self.target_names_l}')

    def _load_settings(self, data_folder):
        self.temporal_adj_matrix = None
        self.adj_matrix = None
        self.settings = {}
        filename = os.path.join(data_folder, 'settings.json')
        if os.path.isfile(filename):
            with open(filename, 'r') as f:
                self.settings = json.load(f)
            self.adj_matrix = torch.Tensor(self.settings['causal_graph'])
            if 'temporal_causal_graph' in self.settings:
                self.temporal_adj_matrix = torch.Tensor(self.settings['temporal_causal_graph'])

    @torch.no_grad()
    def encode_dataset(self, encoder, batch_size=512):
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        encoder.eval()
        encoder.to(device)
        encodings = None
        for idx in tqdm(range(0, self.imgs.shape[0], batch_size), desc='Encoding dataset...', leave=False):
            batch = self.imgs[idx:idx+batch_size].to(device)
            batch = self._prepare_imgs(batch)
            if len(batch.shape) == 5:
                batch = batch.flatten(0, 1)
            batch = encoder(batch)
            if len(self.imgs.shape) == 5:
                batch = batch.unflatten(0, (-1, self.imgs.shape[1]))
            batch = batch.detach().cpu()
            if encodings is None:
                encodings = torch.zeros(self.imgs.shape[:-3] + batch.shape[-1:], dtype=batch.dtype, device='cpu')
            encodings[idx:idx+batch_size] = batch
        self.imgs = encodings
        self.encodings_active = True
        return encodings

    def load_encodings(self, filename):
        self.imgs = torch.load(filename)
        self.encodings_active = True

    def _prepare_imgs(self, imgs):
        if self.encodings_active:
            return imgs
        else:
            imgs = imgs.float() / 255.0
            imgs = imgs * 2.0 - 1.0
            return imgs

    def label_to_img(self, label):
        return (label + 1.0) / 2.0

    def num_labels(self):
        return -1

    def num_vars(self):
        return self.targets.shape[-1]

    def target_names(self):
        return self.target_names_l

    def get_img_width(self):
        return self.imgs.shape[-2]

    def get_inp_channels(self):
        return self.imgs.shape[-3]

    def get_causal_var_info(self):
        return VoronoiDataset.VAR_INFO

    def get_adj_matrix(self):
        return self.adj_matrix

    def get_temporal_adj_matrix(self):
        return self.temporal_adj_matrix

    def __len__(self):
        return self.imgs.shape[0] - self.seq_len + 1

    def __getitem__(self, idx):
        returns = []

        if self.triplet:
            img_pair = self.imgs[idx]
            pos = self.latents[idx]
            target = self.targets[idx]
        else:
            img_pair = self.imgs[idx:idx+self.seq_len]
            pos = self.latents[idx:idx+self.seq_len]
            target = self.targets[idx:idx+self.seq_len-1]

        if self.single_image:
            img_pair = img_pair[0]
            pos = pos[0]
        else:
            returns += [target]
        img_pair = self._prepare_imgs(img_pair)
        returns = [img_pair] + returns

        if self.return_latents:
            returns += [pos]

        return tuple(returns) if len(returns) > 1 else returns[0]


class PinballDataset(data.Dataset):

    VAR_INFO = OrderedDict({
        'ball-x': 'continuous_4',
        'ball-x-vel': 'continuous_8',
        'ball-y': 'continuous_3',
        'ball-y-vel': 'continuous_6',
        'cyl-0-active': 'continuous_1.5',
        'cyl-1-active': 'continuous_1.5',
        'cyl-2-active': 'continuous_1.5',
        'cyl-3-active': 'continuous_1.5',
        'cyl-4-active': 'continuous_1.5',
        'paddle-left-y-pos': 'continuous_1.5',
        'paddle-right-y-pos': 'continuous_1.5',
        'score': 'categ_20'
    })

    def __init__(self, data_folder, split='train', single_image=False, return_latents=False, triplet=False, seq_len=2, causal_vars=None, **kwargs):
        super().__init__()
        filename = split
        if triplet:
            filename += '_triplets'
        self.split_name = filename
        data_file = os.path.join(data_folder, f'{filename}.npz')
        if split.startswith('val') and not os.path.isfile(data_file):
            self.split_name = self.split_name.replace('val', 'test')
            print('[!] WARNING: Could not find a validation dataset. Falling back to the standard test set. Do not use it for selecting the best model!')
            data_file = os.path.join(data_folder, f'{filename.replace("val", "test")}.npz')
        assert os.path.isfile(data_file), f'Could not find ComplexInterventionalPong dataset at {data_file}'

        arr = np.load(data_file)
        self.imgs = torch.from_numpy(arr['images'])
        self.latents = torch.from_numpy(arr['latents'])
        self.targets = torch.from_numpy(arr['targets'])
        self.keys_latents = [key.replace('_', '-') for key in arr['keys'].tolist()]
        if 'keys_targets' in arr:
            self.keys_targets = [key.replace('_', '-') for key in arr['keys_targets'].tolist()]
        else:
            self.keys_targets = self.keys_latents
        self._clean_up_data(causal_vars)

        self.single_image = single_image
        self.return_latents = return_latents
        self.triplet = triplet
        self.encodings_active = False
        self.seq_len = seq_len if not (single_image or triplet) else 1

    def _clean_up_data(self, causal_vars=None):
        if len(self.imgs.shape) == 5:
            self.imgs = self.imgs.permute(0, 1, 4, 2, 3)  # Push channels to PyTorch dimension
        else:
            self.imgs = self.imgs.permute(0, 3, 1, 2)

        all_latents, all_targets = [], []
        target_names = self.keys_targets if causal_vars is None else causal_vars
        keys_var_info = list(PinballDataset.VAR_INFO.keys())
        for key in keys_var_info:
            if key not in self.keys_latents:
                PinballDataset.VAR_INFO.pop(key)
        for i, key in enumerate(self.keys_latents):
            scale = float(PinballDataset.VAR_INFO[key].split('_')[-1])
            if key in ['ball-x', 'ball-y']:
                self.latents[..., i] = self.latents[..., i] / 16 - 1
            elif key in ['ball-x-vel', 'ball-y-vel']:
                self.latents[..., i] = self.latents[..., i] / 8
            elif key.startswith('cyl-'):
                self.latents[..., i] = self.latents[..., i] * 2 - 1
            elif key.startswith('paddle-'):
                self.latents[..., i] = self.latents[..., i] / 2 - 2
            else:
                continue
            self.latents[..., i] *= scale
        self.target_names_l = target_names
        print(f'Using the causal variables {self.target_names_l}')

    @torch.no_grad()
    def encode_dataset(self, encoder, batch_size=512):
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        encoder.eval()
        encoder.to(device)
        encodings = None
        for idx in tqdm(range(0, self.imgs.shape[0], batch_size), desc='Encoding dataset...', leave=False):
            batch = self.imgs[idx:idx+batch_size].to(device)
            batch = self._prepare_imgs(batch)
            if len(batch.shape) == 5:
                batch = batch.flatten(0, 1)
            batch = encoder(batch)
            if len(self.imgs.shape) == 5:
                batch = batch.unflatten(0, (-1, self.imgs.shape[1]))
            batch = batch.detach().cpu()
            if encodings is None:
                encodings = torch.zeros(self.imgs.shape[:-3] + batch.shape[-1:], dtype=batch.dtype, device='cpu')
            encodings[idx:idx+batch_size] = batch
        self.imgs = encodings
        self.encodings_active = True
        return encodings

    def load_encodings(self, filename):
        self.imgs = torch.load(filename)
        self.encodings_active = True

    def _prepare_imgs(self, imgs):
        if self.encodings_active:
            return imgs
        else:
            imgs = imgs.float() / 255.0
            imgs = imgs * 2.0 - 1.0
            return imgs

    def label_to_img(self, label):
        return (label + 1.0) / 2.0

    def num_labels(self):
        return -1

    def num_vars(self):
        return self.targets.shape[-1]

    def target_names(self):
        return self.target_names_l

    def get_img_width(self):
        return self.imgs.shape[-2]

    def get_inp_channels(self):
        return self.imgs.shape[-3]

    def get_causal_var_info(self):
        return PinballDataset.VAR_INFO

    def __len__(self):
        return self.imgs.shape[0] - self.seq_len + 1

    def __getitem__(self, idx):
        returns = []

        if self.triplet:
            img_pair = self.imgs[idx]
            pos = self.latents[idx]
            target = self.targets[idx]
        else:
            img_pair = self.imgs[idx:idx+self.seq_len]
            pos = self.latents[idx:idx+self.seq_len]
            target = self.targets[idx:idx+self.seq_len-1]

        if self.single_image:
            img_pair = img_pair[0]
            pos = pos[0]
        else:
            returns += [target]
        img_pair = self._prepare_imgs(img_pair)
        returns = [img_pair] + returns

        if self.return_latents:
            returns += [pos]

        return tuple(returns) if len(returns) > 1 else returns[0]


# FACTORS_old = ['x_pos', 'y_pos', 'z_pos', 'alpha', 'beta', 'rot_spotlight', 'hue_object', 'hue_spotlight',
#                'hue_background', 'shape']
# COARSE_FACTORS_old = ['pos', 'rot', 'rot_spotlight', 'hue_object', 'hue_spotlight', 'hue_background', 'shape']
# F2SHORT_old = {
#     'pos': 'p',
#     'x_pos': 'x',
#     'y_pos': 'y',
#     'z_pos': 'z',
#     'rot': 'r',
#     'alpha': 'a',
#     'beta': 'b',
#     'rot_spotlight': 'rs',
#     'hue_object': 'ho',
#     'hue_spotlight': 'hs',
#     'hue_background': 'hb',
#     'shape': 's',
#     'trash': 't'
# }
# SHORT2F_old = {v: k for k, v in F2SHORT_old.items()}
# SHORT_FACTORS_old = [F2SHORT_old[f] for f in FACTORS_old]
# SHORT_COARSE_FACTORS_old = [F2SHORT_old[f] for f in COARSE_FACTORS_old]
# SHORT_COARSE_FACTORS_AND_TRASH_old = SHORT_COARSE_FACTORS_old + ['t']
# FINE2COARSE_old = {
#     'x_pos': 'pos',
#     'y_pos': 'pos',
#     'z_pos': 'pos',
#     'alpha': 'rot',
#     'beta': 'rot',
#     'rot_spotlight': 'rot_spotlight',
#     'hue_object': 'hue_object',
#     'hue_spotlight': 'hue_spotlight',
#     'hue_background': 'hue_background',
#     'shape': 'shape',
# }
# SHORT_FINE2COARSE_old = {F2SHORT_old[k]: F2SHORT_old[v] for k, v in FINE2COARSE_old.items()}


def get_DataClass_for_datadir(data_dir):
    if 'ball_in_boxes' in data_dir:
        DataClass = BallInBoxesDataset
    elif 'pong' in data_dir:
        DataClass = InterventionalPongDataset
    elif 'causal3d' in data_dir:
        DataClass = Causal3DDataset
    elif 'voronoi' in data_dir:
        DataClass = VoronoiDataset
    elif 'pinball' in data_dir:
        DataClass = PinballDataset
    else:
        DataClass = Causal3DDataset
    return DataClass
