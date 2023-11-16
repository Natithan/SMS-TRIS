"""
Create a sequence of causal factors for the Temporal Causal3DIdent dataset.
"""

import os

import numpy as np
import argparse
import json
from myutil import Namespace, namespace_to_arglist

factorname2id = {
    'x_pos': 0,
    'y_pos': 1,
    'z_pos': 2,
    'alpha': 3,
    'beta': 4,
    'gamma': 5,
    'rot_spotlight': 6,
    'hue_object': 7,
    'hue_spotlight': 8,
    'hue_background': 9,
    'shape': 10,
    'material': 11
}

# See datasets.Causal3DDataset
# ID_F2P = {
#     'x_pos': ['x_pos', 'beta'],
#     'y_pos': ['y_pos', 'alpha'],
#     'z_pos': ['z_pos', 'alpha'],
#     'alpha': ['alpha', 'hue_background'],
#     'beta': ['beta', 'hue_object'],
#     'rot_spotlight': ['rot_spotlight', 'x_pos', 'y_pos'],
#     'hue_object': ['hue_object', 'shape','hue_spotlight','hue_background'],
#     'hue_spotlight': ['hue_spotlight', 'hue_background'],
#     'hue_background': ['hue_background'],
#     'shape': ['shape'],
# }
# OOD_F2P = {
#     'x_pos': ['x_pos', 'beta'],
#     'y_pos': ['y_pos', 'alpha'],
#     'z_pos': ['z_pos', 'beta'],
#     'alpha': ['alpha', 'hue_object'],
#     'beta': ['beta', 'hue_background'],
#     'rot_spotlight': ['rot_spotlight', 'y_pos', 'z_pos'],
#     'hue_object': ['hue_object', 'shape'],
#     'hue_spotlight': ['hue_spotlight', 'hue_background'],
#     'hue_background': ['hue_background', 'hue_object'],
#     'shape': ['shape'],
# }


def angle_mean(angles):
    """
    Returns the mean of angles while respecting circular nature of the angle values
    See https://en.wikipedia.org/wiki/Circular_mean
    """
    angles = np.array(angles)
    avg_angle = np.arctan2(np.sin(angles).mean(),
                           np.cos(angles).mean())
    return avg_angle


def correct_angle(val):
    """
    Bringing angle back to [0, 2pi) value space
    """
    while val >= 2 * np.pi:
        val -= 2 * np.pi
    while val < 0.0:
        val += 2 * np.pi
    return val


def perform_time_step(latents_t, noise_t, shape_t, interventions, intv_vals, exclude_vars, object_map, ood, stds,
                      instantaneous=False):
    """
    Given the latents of a time step t, sample new latents for time step t+1
    """
    if not instantaneous:
        goals = get_goals(exclude_vars, latents_t, shape_t, object_map, ood)

        new_vals = latents_t * 0.0
        for i in range(goals.shape[0]):
            new_vals[i] = _take_step(latents_t[i], goals[i], noise_t[i], angle=(i in [3, 4, 5, 6, 7, 8, 9]), stds=stds)
    else:
        new_vals = latents_t * 0.0

        def _take_step_with_intv(index, goal):
            if interventions[index]:
                new_vals[index] = intv_vals[index]
            else:
                new_vals[index] = _take_step(latents_t[index], goal, noise_t[index],
                                             angle=(index in [3, 4, 5, 6, 7, 8, 9]), stds=stds)

        # Background hue
        _take_step_with_intv(9, latents_t[9])
        # Spotlight hue
        _take_step_with_intv(8, 2 * np.pi - new_vals[9])
        # Object hue
        if 10 not in exclude_vars:
            shape = object_map[shape_t[0]]
            if shape == 0:  # Teapot
                goal = 0.0
            elif shape == 1:  # Armadillo
                goal = 2 * np.pi * (1.0 / 5.0)
            elif shape == 2:  # Bunny / Hare
                goal = angle_mean([new_vals[8], new_vals[9]])
            elif shape == 3:  # Cow
                goal = 2 * np.pi * (2.0 / 5.0)
            elif shape == 4:  # Dragon
                goal = np.pi + angle_mean([new_vals[8], new_vals[9]])
            elif shape == 5:  # Head
                goal = 2 * np.pi * (3.0 / 5.0)
            elif shape == 6:  # Horse
                goal = 2 * np.pi * (4.0 / 5.0)
            goal = correct_angle(goal)
        else:
            goal = latents_t[7]
        _take_step_with_intv(7, goal)
        # Rotation alpha and beta
        _take_step_with_intv(3, new_vals[9])
        _take_step_with_intv(4, new_vals[7])
        # position x, y and z
        _take_step_with_intv(0, 1.5 * np.sin(new_vals[4]))
        _take_step_with_intv(1, 1.5 * np.sin(new_vals[3]))
        _take_step_with_intv(2, 1.5 * np.cos(new_vals[3]))
        # Spotlight rotation
        _take_step_with_intv(6, correct_angle(np.arctan2(new_vals[0], new_vals[1])))
    # Return everything
    return new_vals


def get_goals(exclude_vars, latents_t, shape_t, object_map, ood):
    goals = np.zeros_like(latents_t)

    # Old version
    # x position
    # goals[0] = 1.5 * np.sin(latents_t[4])
    # # y-z position
    # goals[1] = 1.5 * np.sin(latents_t[3])
    # goals[2] = 1.5 * np.cos(latents_t[3])
    # # alpha and beta
    # goals[3] = latents_t[9]
    # goals[4] = latents_t[7]
    # # spot light - hue opposite of background
    # goals[8] = 2 * np.pi - latents_t[9]
    # # spot light - rotation opposite of x-y position
    # goals[6] = correct_angle(np.arctan2(latents_t[0], latents_t[1]))
    # # hue object - dependent on the shape
    # if 10 not in exclude_vars:
    #     shape = object_map[shape_t[0]]
    #     goals[7] = get_obj_hue_goal(spotlight_hue=latents_t[8], background_hue=latents_t[9], shape=shape)
    # else:
    #     goals[7] = latents_t[7]
    # # hue background - independent of everything
    # goals[9] = latents_t[9]

    # Nathan version
    f2id = factorname2id
    mechs = {
        'x_pos': lambda l: 1.5 * np.sin(l[f2id['beta']]),
        'y_pos': lambda l: 1.5 * np.sin(l[f2id['alpha']]),
        'z_pos': lambda l: 1.5 * np.cos(l[f2id['alpha']]),
        'alpha': lambda l: l[f2id['hue_background']],
        'beta': lambda l: l[f2id['hue_object']],
        'rot_spotlight': lambda l: correct_angle(np.arctan2(l[f2id['x_pos']], l[f2id['y_pos']])),
        'hue_spotlight': lambda l: 2 * np.pi - l[f2id['hue_background']],
        'hue_object': lambda l:
        l[f2id['hue_object']] if f2id['shape'] in exclude_vars else
        get_obj_hue_goal(l[f2id['hue_spotlight']], l[f2id['hue_background']], object_map[shape_t[0]]),
        'hue_background': lambda l: l[f2id['hue_background']]
    }  # gamma, shape & material are excluded

    ood_mechs = {
        'x_pos': lambda l: 1.5 * np.cos(l[f2id['beta']]),
        # Change transformation: x=f(1.5*sin(beta),x,e_x) -> x=f(1.5*cos(beta),x,e_x)
        'y_pos': lambda l: -1.5 * np.cos(l[f2id['alpha']]),
        # Change transformation: y=f(1.5*sin(alpha),y,e_y)  -> y=f(-1.5*cos(alpha),y,e_y)
        'z_pos': lambda l: 1.5 * np.sin(l[f2id['beta']]),
        # Change transformation and domain: z=f(1.5*cos(alpha),z,e_z) -> z=f(1.5*sin(beta),z,e_z)

        # Flip domains of alpha and beta mechanisms compared to og mechanisms
        'alpha': lambda l: l[f2id['hue_object']],
        # alpha=f(hue_background,alpha,e_alpha) -> alpha=f(hue_object,alpha,e_alpha)
        'beta': lambda l: l[f2id['hue_background']],
        # beta=f(hue_object,beta,e_beta) -> beta=f(hue_background,beta,e_beta)

        'rot_spotlight': lambda l: correct_angle(np.arctan2(l[f2id['y_pos']], l[f2id['z_pos']])),
        # rot_spotlight=f(atan(x_pos,y_pos),rot_spotlight,e_rot_spotlight) -> rot_spotlight=f(atan(y_pos,z_pos),rot_spotlight,e_rot_spotlight)
        # flip inputs, and change one of them to z_pos
        'hue_spotlight': lambda l: 2 * np.pi - l[f2id['hue_background']] * l[f2id['hue_spotlight']],
        # hue_spotlight=f(2*pi - hue_background,hue_spotlight,e_hue_spotlight) -> hue_spotlight=f(2*pi - hue_background*hue_spotlight,hue_spotlight,e_hue_spotlight)
        # Keep domain, but change nonlinearly
        'hue_object': lambda l:
        l[f2id['hue_object']] + l[f2id['hue_object']] - np.pi if f2id[
                                                                     'shape'] not in exclude_vars else  # change domain
        get_obj_hue_goal(l[f2id['hue_spotlight']], l[f2id['hue_background']], object_map[shape_t[0]], ood=True),
        'hue_background': lambda l: l[f2id['hue_background']] + l[f2id['hue_object']]  # Change domain
    }
    for factor_name, factor_mech in mechs.items():
        fid = factorname2id[factor_name]
        goals[fid] = factor_mech(latents_t) if factor_name not in ood else ood_mechs[factor_name](latents_t)

    return goals


def get_obj_hue_goal(spotlight_hue, background_hue, shape, ood=False):
    if shape == 0:  # Teapot
        result = 0.0 if not ood else np.pi  # Just change
    elif shape == 1:  # Armadillo
        result = 2 * np.pi * (1.0 / 5.0) if not ood else 0.0  # Flip with teapot
    elif shape == 2:  # Bunny / Hare
        result = angle_mean([spotlight_hue, background_hue]) if not ood else np.pi + angle_mean(
            [spotlight_hue, background_hue])  # Flip with dragon
    elif shape == 3:  # Cow
        result = 2 * np.pi * (2.0 / 5.0) if not ood else 2 * np.pi * (3.0 / 5.0)  # Flip with head
    elif shape == 4:  # Dragon
        result = np.pi + angle_mean([spotlight_hue, background_hue]) if not ood else angle_mean(
            [spotlight_hue, background_hue])  # Flip with bunny
    elif shape == 5:  # Head
        result = 2 * np.pi * (3.0 / 5.0) if not ood else 2 * np.pi * (2.0 / 5.0)  # Flip with cow
    elif shape == 6:  # Horse
        result = 2 * np.pi * (4.0 / 5.0) if not ood else 2 * np.pi * (4.5 / 5.0)  # Just change
    return correct_angle(result)


def _take_step(current_val, goal_val, noise, stds, angle=False):
    """
    Given a current value and a 'goal' value determined by the parents of a causal variable,
    determine the next value. For linear continuous values, this corresponds to the average
    of the current and goal value, with additive noise.
    """
    step = (goal_val - current_val) / 2.0
    # For angles, we need to respect the circular nature of the values
    if angle:
        if abs(goal_val - current_val) > np.pi:
            if step < 0.0:
                step += np.pi
            elif step > 0.0:
                step -= np.pi
    next_val = current_val + step + noise
    # Sample from truncated Gaussian for x-y-z position
    while not angle and abs(next_val) > 2:
        next_val = current_val + step + stds[0] * np.random.randn(1, )
    return next_val


def generate_latents(args: Namespace=None):
    # region argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_points', default=10000, type=int,
                        help='Number of samples/data points to generate.')
    parser.add_argument('--output_folder', required=True, type=str,
                        help='Folder to save the exported latents to.')
    parser.add_argument('--split', type=str,
                        help='train, val or test', default='train')  # Nathan
    parser.add_argument('--coarse_vars', action='store_true',
                        help='If True, the positions and rotations are considered as multidimensional causal factors. Set to True for the experiments in the paper.')
    parser.add_argument('--int_probs', type=float, default=-1,
                        help='Probability with which an intervention is performed. -1 means we use 1/num_vars+1.')
    parser.add_argument('--single_targets', action='store_true',
                        help='If True, single-target interventions are used instead of independent Bernoulli samples.')
    parser.add_argument('--group_targets', action='store_true',
                        help='If True, grouped interventions as presented in the paper appendix are used.')
    parser.add_argument('--default_object', type=int, default=0,
                        help='The index of the shape, if a single shape dataset should be generated.')
    parser.add_argument('--num_shapes', type=int, default=1,
                        help='Number of shapes to use for generating the dataset. Set to 7 for generating dataset with all shapes.')
    parser.add_argument('--exclude_objects', type=int, nargs='+', default=None,
                        help='List of objects to exclude for generating the dataset, e.g. to test the generalization to unseen shapes.')
    parser.add_argument('--instantaneous', action='store_true',
                        help='If True, uses instantaneous effects instead of temporally shifted.')
    parser.add_argument('--seed', type=int, default=-1,
                        help='Seed to set for deterministic output.')
    parser.add_argument('--ood', type=str, nargs='+',
                        default=None)  # Nathan: which factor mechanisms to replace by an ood mechanism
    parser.add_argument('--iid_pairs', action='store_true',
                        help='If True, instead of a stream of frames, we create a bunch of frame pairs with independent start frames.')
    #endregion
    # if args is not None, use those instead of the command line arguments
    if args is not None:
        args_as_list = namespace_to_arglist(args)
        args = parser.parse_args(args_as_list)
    else:
        args = parser.parse_args()

    print("args.exclude_objects", args.exclude_objects)
    if args.ood is None:
        args.ood = []
    print('Parsed arguments:', args)
    # out_folder = os.path.join(args.output_folder, f"{args.n_points}_points","_".join(args.ood))
    out_folder = args.output_folder
    print("Out folder latents:", out_folder)
    os.makedirs(out_folder, exist_ok=True)

    """
    We aim at generating a numpy array with all causal factor dimensions over time.
    Render internally assumes the variables form these value ranges:
    
    object-based causal factors:
        0. x position in [-2, +2]
        1. y position in [-2, +2]
        2. z position in [-2, +2]
        3. alpha rotation in [0, 2pi]
        4. beta rotation in [0, 2pi]
        5. gamma rotation in [0, 2pi]
        6. theta spot light in [0, 2pi]
        7. hue object in [0, 2pi]
        8. hue spot light in [0, 2pi]
    
    scene-based causal factors:
        9. hue background in [0, 2pi]

    additional ones:
        10. shapes in [0 (Teapot), 1 (Armadillo), 2 (Hare), 3 (Cow), 4 (Dragon), 5 (Head), 6 (Horse)]
        11. material in [0 (Rubber), 1 (Crystal), 2 (MyMetal)] - Not used in experiments for the paper
    """
    seed_add = {
        'train': 0,
        'val': 1,
        'test': 2
    }
    if args.seed < 0:
        np.random.seed(42 + args.n_points + seed_add[args.split])
    else:
        np.random.seed(args.seed + seed_add[args.split])

    # Constants that are helpful for processing all causal variables above in one array
    position_range = 2
    n_angular_variables = 4
    n_color_variables = 3
    n_non_angular_variables = 3
    n_shape_variables = 2
    num_cont_vars = n_angular_variables + n_non_angular_variables + n_color_variables
    num_vars = num_cont_vars + n_shape_variables

    # Render variables that are ignored/not considered as causal variables.
    # We exclude the third rotation since it violates the bijective observation function assumption
    exclude_vars = [5, 11]

    # Determine the object shapes we want to use
    exclude_objects = args.exclude_objects
    if exclude_objects is None:
        exclude_objects = []
    if args.num_shapes <= 1:
        exclude_vars += [10]
    num_categs = [args.num_shapes - len(exclude_objects), 1]
    object_map = [i for i in range(args.num_shapes) if i not in exclude_objects]

    # Set intervention probability 
    if args.int_probs >= 0:
        int_probs = args.int_probs
    else:
        int_probs = (1. / (num_vars - len(exclude_vars) - (4 if args.coarse_vars else 0) + 1))
    # if args.iid_pairs:
    #     print("Setting int_probs to 0.0 for iid pairs")
    #     int_probs = 0.0

    # Set standard deviation for distributions over time
    stds = np.array([0.1] * n_non_angular_variables + [0.15] * n_angular_variables + [0.15] * n_color_variables)
    shape_stds = np.array([0.05, 0.0])

    # Create random values for all causal factors that represent the samples from a potential intervention.
    # It is faster to sample them at once here instead of at each time step
    random_latents = get_random_latents(args, num_cont_vars, position_range)
    random_shapes = get_random_shapes(args, n_shape_variables, num_categs)

    # Create noise for transitions
    random_trans = np.random.randn(args.n_points - 1, num_cont_vars)
    random_trans_shapes = np.random.rand(args.n_points - 1, n_shape_variables)

    # Initialize array for all latents
    latents = np.zeros((args.n_points, num_cont_vars), dtype=np.float32)
    shape_latents = np.zeros((args.n_points, n_shape_variables), dtype=np.uint8)
    # Set starting point
    latents[0] = random_latents[0]
    shape_latents[0] = random_shapes[0]

    # Sample intervention targets
    interventions = np.random.binomial(n=1, p=int_probs,
                                       size=(args.n_points - 1, num_vars))
    interventions[:, exclude_vars] = 0
    if int_probs < 1.0:
        if args.single_targets:  # Convert the random binomial samples to single target interventions
            if args.coarse_vars:
                interventions[:, 1:3] = 0
                interventions[:, 4:6] = 0
            cont_prob = interventions * np.random.rand(*interventions.shape)
            one_hot = np.zeros_like(interventions)
            one_hot[np.arange(one_hot.shape[0]), cont_prob.argmax(axis=-1)] = 1
            interventions = interventions * one_hot
        elif args.group_targets:  # Convert the random binomial samples to group interventions
            if args.coarse_vars:
                groups = np.zeros((5, interventions.shape[-1]), interventions.dtype)
                groups[0, :3] = 1  # Position
                groups[1, 3:6] = 1  # Object rotation
                groups[2, 6] = 1  # Joint rot_s and hue_s
                groups[2, 8] = 1  # Joint rot_s and hue_s
                groups[3, 7:9] = 1  # Joint hue_s and hue_obj
                groups[4, 7] = 1  # Joint hue_obj and hue_b
                groups[4, 9] = 1  # Joint hue_obj and hue_b
                obs_prob = (1 - int_probs) ** 6
                if 10 not in exclude_vars:  # Assumed to not include shapes as random variable
                    raise NotImplementedError
            else:  # Group targets for fine variables can be defined here if needed
                raise NotImplementedError
            group_sel = np.random.randint(groups.shape[0], size=(interventions.shape[0],))
            obs_int = np.random.binomial(n=1, p=1 - obs_prob,
                                         size=(interventions.shape[0],))
            interventions = groups[group_sel] * obs_int[:, None]
    if args.coarse_vars:
        interventions[:, 0:3] = interventions[:, :1]
        interventions[:, 3:6] = interventions[:, 3:4]
    np.save(os.path.join(out_folder, f'interventions_{args.split}.npy'), interventions)

    if args.iid_pairs:
        in_shape_latents = get_random_shapes(args, n_shape_variables, num_categs)[
                           :-1].astype(np.uint8)  # make new random shapes for the in_latents
        in_latents = get_random_latents(args, num_cont_vars, position_range)[
                     :-1].astype(np.float32)  # make new random latents for the in_latents
        shape_latents = shape_latents[:-1]
        latents = latents[:-1]
    # Iterate over all time steps and perform transitions with interventions
    for i in range(1, args.n_points):
        # Perform time step for shape latents
        new_shape_latent = np.where(
            np.logical_or(interventions[i - 1, num_cont_vars:], random_trans_shapes[i - 1] < shape_stds),
            # Intervention or noise both entail sampling from random_shapes
            (random_shapes[i] if not args.iid_pairs else random_shapes[i - 1]),
            (shape_latents[i - 1] if not args.iid_pairs else in_shape_latents[i - 1]))
        # Perform time step for other latents
        new_latent = perform_time_step(latents[i - 1] if not args.iid_pairs else in_latents[i - 1],
                                       random_trans[i - 1] * stds,
                                       shape_latents[(i - 1) if not args.instantaneous else i] if not args.iid_pairs else in_shape_latents[i - 1],
                                       interventions=interventions[i - 1, :num_cont_vars],
                                       intv_vals=random_latents[i],
                                       exclude_vars=exclude_vars,
                                       instantaneous=args.instantaneous,
                                       object_map=object_map,
                                       ood=args.ood,
                                       stds=stds)
        new_latent = np.where(interventions[i - 1, :num_cont_vars], random_latents[i], new_latent)
        for j in range(n_non_angular_variables):
            while abs(new_latent[
                          j]) > 2:  # Nathan: so if accidentally the position is too far, this replaces the natural mechanism with a random jump?? Also, isn't this already taken care of at the end of _take_step?
                new_latent[j] = latents[i - 1, j] + np.random.randn(1, ) * stds[j]
        new_latent[n_non_angular_variables:] = np.fmod(new_latent[n_non_angular_variables:], 2 * np.pi) + 2 * np.pi * (
                new_latent[n_non_angular_variables:] < 0)  # moves values between 0 and 2pi
        if args.iid_pairs:
            shape_latents[i - 1] = new_shape_latent
            latents[i - 1] = new_latent
        else:
            shape_latents[i] = new_shape_latent
            latents[i] = new_latent

    # Map object indices back to exclude specified object indices
    for i in range(shape_latents.shape[0]):  # aka args.n_points
        shape_latents[i, 0] = object_map[shape_latents[i, 0]]
        if args.iid_pairs:
            in_shape_latents[i, 0] = object_map[in_shape_latents[i, 0]]

    # Set excluded variables to constant factors
    for v in exclude_vars:
        if v < latents.shape[-1]:
            latents[:, v] = 0.0
            if args.iid_pairs:
                in_latents[:, v] = 0.0
        else:
            if v - num_cont_vars == 0:
                shape_latents[:, v - num_cont_vars] = args.default_object
                if args.iid_pairs:
                    in_shape_latents[:, v - num_cont_vars] = args.default_object
            else:
                shape_latents[:, v - num_cont_vars] = 0
                if args.iid_pairs:
                    in_shape_latents[:, v - num_cont_vars] = 0

    # Save hyperparameters for transparency
    with open(os.path.join(out_folder, 'hparams.json'), 'w') as f:
        hparams = {
            'position_range': position_range,
            'n_angular_variables': n_angular_variables,
            'n_non_angular_variables': n_non_angular_variables,
            'n_color_variables': n_color_variables,
            'n_shape_variables': n_shape_variables,
            'exclude_vars': exclude_vars,
            'num_categs': num_categs,
            'int_probs': int_probs,
            'stds': stds.tolist(),
            'coarse_vars': args.coarse_vars,
            'use_spotlight': (6 not in exclude_vars),
            'default_object': args.default_object,
            'num_shapes': args.num_shapes
        }
        json.dump(hparams, f, indent=4)
    # Save generated latents
    if not args.iid_pairs:
        np.save(os.path.join(out_folder, f'latents_{args.split}.npy'), latents)
        np.save(os.path.join(out_folder, f'shape_latents_{args.split}.npy'), shape_latents)
    else:
        for time, type, lat in ([
            ('in', 'shape_', in_shape_latents),
            ('in', '', in_latents),
            ('out', 'shape_', shape_latents),
            ('out', '', latents)
        ]):
            np.save(os.path.join(out_folder, f'{time}_{type}latents_{args.split}.npy'), lat)


def get_random_shapes(args, n_shape_variables, num_categs):
    return np.stack([np.random.randint(num_categs[i], size=(args.n_points)) for i in range(n_shape_variables)],
                    axis=-1)


def get_random_latents(args, num_cont_vars, position_range):
    random_latents = np.random.rand(args.n_points, num_cont_vars)
    random_latents[:, :3] = (random_latents[:, :3] - 0.5) * 2 * position_range
    random_latents[:, 3:] *= 2 * np.pi
    return random_latents


if __name__ == '__main__':
    generate_latents()