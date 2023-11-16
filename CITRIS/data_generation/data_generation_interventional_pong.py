"""
Create the Interventional Pong dataset including rendering.
"""
import operator
import time
import numpy as np
import matplotlib
matplotlib.use('Agg') # Otherwise if DISPLAY is not None, it uses TKAgg, which is muy slow
import matplotlib.pyplot as plt
import os
from imageio import imread
import json
from multiprocessing import Pool
import time
from glob import glob
from argparse import ArgumentParser


def paddle_step(paddle_y, ball_y, settings, intervention=False):
    """
    Sample the next position of a paddle.
    """
    if not intervention:
        step = min(abs(paddle_y - ball_y), settings['paddle_max_step'])
        if paddle_y > ball_y:
            step *= -1
    else:
        step = settings['paddle_max_step']
        if np.random.uniform() > 0.5:
            step = -step
    new_paddle_y = paddle_y + step + np.random.randn() * settings['paddle_step_noise']
    return new_paddle_y


def sample_ball_vel_dir(settings):
    """
    Samples a ball velocity direction. Can be restricted if needed.
    """
    return np.random.uniform(0, 2*np.pi)


def mod_angle(angle):
    """
    Modulo for angles.
    """
    while angle < 0.0:
        angle += 2*np.pi
    while angle > 2*np.pi:
        angle -= 2*np.pi
    return angle


def angle_flip(angle, axis='y'):
    """
    Determines the new velocity direction when the ball hits the borders or paddles.
    """
    if axis == 'x':
        angle = mod_angle(np.pi - angle)
    elif axis == 'y':
        angle = mod_angle(angle + 0.5 * np.pi)
        angle = mod_angle(np.pi - angle)
        angle = mod_angle(angle - 0.5 * np.pi)
    return angle


def ball_collision(paddle_tag, new_time_step, prev_time_step, settings):
    """
    Check whether the ball collides with a paddle.
    """
    paddle_x, paddle_y = settings[paddle_tag+'_x'], prev_time_step[paddle_tag+'_y']
    paddle_height = settings['paddle_height']
    paddle_width = settings['paddle_width']
    ball_x_center = new_time_step['ball_x']
    ball_y = new_time_step['ball_y']
    if paddle_tag.endswith('right'):
        ball_x_outer = ball_x_center + settings['ball_radius']
    else:
        ball_x_outer = ball_x_center - settings['ball_radius']
    
    if ball_y > paddle_y + paddle_height / 2.0 or ball_y < paddle_y - paddle_height / 2.0:
        return False

    if paddle_tag.endswith('right'):
        if ball_x_center < paddle_x + paddle_width / 2.0 and ball_x_outer > paddle_x - paddle_width / 2.0:
            return True
        else:
            return False
    if paddle_tag.endswith('left'):
        if ball_x_center > paddle_x - paddle_width / 2.0 and ball_x_outer < paddle_x + paddle_width / 2.0:
            return True
        else:
            return False
    return False


def hard_limit(v, v_min, v_max):
    """
    Limit a value 'v' between v_min and v_max.
    """
    return max(min(v, v_max), v_min)


def put_in_boundaries(time_step, settings):
    """
    For all variables, make sure they are in their corresponding boundaries.
    """
    for key in time_step:
        if key == 'ball_vel_dir':
            time_step[key] = mod_angle(time_step[key])
        elif (key + '_min') in settings and (key + '_max') in settings:
            time_step[key] = hard_limit(time_step[key],
                                        v_min=settings[key+'_min'],
                                        v_max=settings[key+'_max'])
    return time_step


def next_step_interventions(prev_time_step, settings, ball_reset=False):
    """
    Sample an intervention value for every causal variable.
    """
    new_time_step = dict()

    if not ball_reset:
        new_time_step['paddle_left_y'] = paddle_step(prev_time_step['paddle_left_y'], prev_time_step['ball_y'], settings, intervention=True)
        new_time_step['paddle_right_y'] = paddle_step(prev_time_step['paddle_right_y'], prev_time_step['ball_y'], settings, intervention=True)
    else:
        new_time_step['paddle_left_y'] = np.random.uniform(settings['center_point_y']*0.25, settings['center_point_y']*1.75)
        new_time_step['paddle_right_y'] = np.random.uniform(settings['center_point_y']*0.25, settings['center_point_y']*1.75)
    new_time_step['ball_x'] = np.random.uniform(settings['ball_x_min_sample'], settings['ball_x_max_sample'])
    new_time_step['ball_y'] = np.random.uniform(settings['ball_y_min'], settings['ball_y_max'])
    new_time_step['ball_vel_dir'] = sample_ball_vel_dir(settings)
    new_time_step = put_in_boundaries(new_time_step, settings)
    return new_time_step


def next_step_regular(prev_time_step, settings):
    """
    Perform transition from one time step to the next one without any interventions.
    """
    new_time_step = dict()

    new_time_step['paddle_left_y'] = paddle_step(prev_time_step['paddle_left_y'], prev_time_step['ball_y'], settings)
    new_time_step['paddle_right_y'] = paddle_step(prev_time_step['paddle_right_y'], prev_time_step['ball_y'], settings)

    vel_y = np.cos(prev_time_step['ball_vel_dir']) * prev_time_step['ball_vel_magn']
    vel_x = np.sin(prev_time_step['ball_vel_dir']) * prev_time_step['ball_vel_magn']
    new_time_step['ball_x'] = prev_time_step['ball_x'] + vel_x

    point_left, point_right, ball_reset = False, False, False
    if new_time_step['ball_x'] < settings['ball_x_min_point']:
        point_right = True
        ball_reset = True
    elif new_time_step['ball_x'] > settings['ball_x_max_point']:
        point_left = True
        ball_reset = True

    if ball_reset:
        new_time_step['ball_x'] = settings['center_point_x']
        new_time_step['ball_y'] = settings['center_point_y']
        new_time_step['ball_vel_dir'] = sample_ball_vel_dir(settings)
        new_time_step['paddle_left_y'] = np.random.uniform(settings['center_point_y']*0.25, settings['center_point_y']*1.75)
        new_time_step['paddle_right_y'] = np.random.uniform(settings['center_point_y']*0.25, settings['center_point_y']*1.75)
    else:
        # Deterministic ball dynamics
        new_time_step['ball_y'] = prev_time_step['ball_y'] + vel_y
        new_time_step['ball_vel_dir'] = prev_time_step['ball_vel_dir']
        # Collisions - Wall (top / bottom)
        if new_time_step['ball_y'] > settings['ball_y_max']:
            new_time_step['ball_y'] = settings['ball_y_max'] - (new_time_step['ball_y'] - settings['ball_y_max'])
            new_time_step['ball_vel_dir'] = angle_flip(new_time_step['ball_vel_dir'], axis='x')
        elif new_time_step['ball_y'] < settings['ball_y_min']:
            new_time_step['ball_y'] = settings['ball_y_min'] - (new_time_step['ball_y'] - settings['ball_y_min'])
            new_time_step['ball_vel_dir'] = angle_flip(new_time_step['ball_vel_dir'], axis='x')
        # Collision - Paddles
        if ball_collision('paddle_left', new_time_step, prev_time_step, settings):
            new_time_step['ball_x'] = (settings['paddle_left_x'] + settings['paddle_width']/2.0) * 2 - (new_time_step['ball_x'] - settings['ball_radius']*2)
            new_time_step['ball_vel_dir'] = angle_flip(new_time_step['ball_vel_dir'], axis='y')
        elif ball_collision('paddle_right', new_time_step, prev_time_step, settings):
            new_time_step['ball_x'] = (settings['paddle_right_x'] - settings['paddle_width']/2.0) * 2 - (new_time_step['ball_x'] + settings['ball_radius']*2)
            new_time_step['ball_vel_dir'] = angle_flip(new_time_step['ball_vel_dir'], axis='y')
        # Noise addition
        new_time_step['ball_x'] += np.random.randn() * settings['ball_x_noise']
        new_time_step['ball_y'] += np.random.randn() * settings['ball_y_noise']
        new_time_step['ball_vel_dir'] += np.random.randn() * settings['ball_vel_dir_noise']

    new_time_step['score_left'] = prev_time_step['score_left'] + int(point_left)
    new_time_step['score_right'] = prev_time_step['score_right'] + int(point_right)
    if max(new_time_step['score_right'], new_time_step['score_left']) >= settings['max_points']:
        new_time_step['score_left'] = 0
        new_time_step['score_right'] = 0

    new_time_step['ball_vel_magn'] = prev_time_step['ball_vel_magn']  # Constant velocity magnitude

    new_time_step = put_in_boundaries(new_time_step, settings)
    return new_time_step, ball_reset

def ball_x_after_paddle_collision(dir, ball_x_latest, settings):
    if dir == 'left':
        return (settings['paddle_left_x'] + settings['paddle_width']/2.0) * 2 - (ball_x_latest - settings['ball_radius']*2)
    elif dir == 'right':
        return (settings['paddle_right_x'] - settings['paddle_width']/2.0) * 2 - (ball_x_latest + settings['ball_radius']*2)
    else:
        raise ValueError(f'Invalid paddle direction {dir}.')



def paddle_collision(paddle_tag, ball_x_center, ball_y_center, prev_time_step, settings):
    """
    Check whether the ball collides with a paddle.
    """
    paddle_x, paddle_y = settings[paddle_tag + '_x'], prev_time_step[paddle_tag + '_y']
    paddle_height = settings['paddle_height']
    paddle_width = settings['paddle_width']
    if paddle_tag.endswith('right'):
        ball_x_outer = ball_x_center + settings['ball_radius']
    else:
        ball_x_outer = ball_x_center - settings['ball_radius']

    if ball_y_center > paddle_y + paddle_height / 2.0 or ball_y_center < paddle_y - paddle_height / 2.0:
        return False

    if paddle_tag.endswith('right'):
        if ball_x_center < paddle_x + paddle_width / 2.0 and ball_x_outer > paddle_x - paddle_width / 2.0:
            return True
        else:
            return False
    if paddle_tag.endswith('left'):
        if ball_x_center > paddle_x - paddle_width / 2.0 and ball_x_outer < paddle_x + paddle_width / 2.0:
            return True
        else:
            return False
    return False

def wall_collision(ball_y, edge, settings):
    if edge == 'max':
        return ball_y > settings['ball_y_max']
    elif edge == 'min':
        return ball_y < settings['ball_y_min']
    else:
        raise ValueError(f'Unknown edge: {edge}')

def alt_next_step_regular(prev_time_step, settings):
    """
    Perform transition from one time step to the next one without any interventions.
    """
    id_new_time_step = dict()
    ood_latest_step = dict()
    ball_vel_magn_latest = prev_time_step['ball_vel_magn']
    vel_y_prev, vel_x_prev = [project(prev_time_step['ball_vel_dir']) * prev_time_step['ball_vel_magn'] for project in (np.cos, np.sin)]
    ball_x_latest = prev_time_step['ball_x'] + vel_x_prev
    point_left, point_right = check_point(ball_x_latest, settings)
    ball_reset = point_left or point_right
    if ball_reset:
        id_new_time_step['ball_x'] = settings['center_point_x']
        id_new_time_step['ball_y'] = settings['center_point_y']
        id_new_time_step['ball_vel_dir'] = sample_ball_vel_dir(settings)
        id_new_time_step['paddle_left_y'] = np.random.uniform(settings['center_point_y']*0.25, settings['center_point_y']*1.75)
        id_new_time_step['paddle_right_y'] = np.random.uniform(settings['center_point_y']*0.25, settings['center_point_y']*1.75)
        for key in ['ball_x','ball_y','ball_vel_dir','paddle_left_y','paddle_right_y']:
            ood_latest_step[key] = id_new_time_step[key]
    else:
        for paddle_dir in ('left', 'right'):
            id_new_time_step[f'paddle_{paddle_dir}_y'] = paddle_step(prev_time_step[f'paddle_{paddle_dir}_y'], prev_time_step['ball_y'], settings)

        # Rest of deterministic ball dynamics
        ball_y_latest = prev_time_step['ball_y'] + vel_y_prev
        ball_vel_dir_latest = prev_time_step['ball_vel_dir']
        # Collisions - Wall (top / bottom)
        for edge in ('max', 'min'):
            if wall_collision(ball_y_latest, edge, settings):
                ball_y_latest = settings[f'ball_y_{edge}'] - (ball_y_latest - settings[f'ball_y_{edge}'])
                ball_vel_dir_latest = angle_flip(ball_vel_dir_latest, axis='x')
                break
        # Collision - Paddles
        ood_latest_step[f'paddle_left_y'], ood_latest_step[f'paddle_right_y'] = id_new_time_step[f'paddle_left_y'], id_new_time_step[f'paddle_right_y']
        ood_latest_step['ball_vel_dir'] = ball_vel_dir_latest
        for paddle_dir in ('left', 'right'):
            if paddle_collision(f'paddle_{paddle_dir}', ball_x_latest, ball_y_latest, prev_time_step, settings):
                ball_x_latest = ball_x_after_paddle_collision(paddle_dir, ball_x_latest, settings)
                ball_vel_dir_latest = angle_flip(ball_vel_dir_latest, axis='y')
                #region ball_vel_dir OOD change: when paddle collision, now also angle flip around x-axis
                ood_latest_step['ball_vel_dir'] = ball_vel_dir_latest
                ood_latest_step['ball_vel_dir'] = angle_flip(ood_latest_step['ball_vel_dir'], axis='x')
                #endregion

                # region paddle OOD change: paddle teleports a halfway up or a halfway down depending on if the paddle is in the lower or upper half of the screen
                inter_wall_distance = settings['ball_y_max'] - settings['ball_y_min']
                if ood_latest_step[f'paddle_{paddle_dir}_y'] < settings['center_point_y']:
                    ood_latest_step[f'paddle_{paddle_dir}_y'] += inter_wall_distance / 2
                else:
                    ood_latest_step[f'paddle_{paddle_dir}_y'] -= inter_wall_distance / 2
                # endregion
                break
        # region ball_x OOD change: if ball_x is in a quarter of the inter-paddle-distance, it teleports to the other quarter
        ood_latest_step['ball_x'] = ball_x_latest
        inter_paddle_distance = settings['paddle_right_x'] - settings['paddle_left_x']
        left_quart_x = settings['paddle_left_x'] + inter_paddle_distance / 4
        right_quart_x = settings['paddle_right_x'] - inter_paddle_distance / 4
        # from left to right, if ball is at left quarter, and velocity is to the right
        if left_quart_x < ball_x_latest < right_quart_x:
            if (ood_latest_step['ball_x'] - prev_time_step['ball_x']) > 0:
                ood_latest_step['ball_x'] = right_quart_x
            else:
                ood_latest_step['ball_x'] = left_quart_x
        #endregion
        # region ball_y OOD change: if ball_y is in 40% of the inter-wall-distance, it teleports to the other 40%
        ood_latest_step['ball_y'] = ball_y_latest
        inter_wall_distance = settings['ball_y_max'] - settings['ball_y_min']
        top_portal_y = settings['ball_y_max'] - inter_wall_distance * 0.40
        bottom_portal_y = settings['ball_y_min'] + inter_wall_distance * 0.40
        # from top to bottom, if ball is at top portal, and velocity is to the bottom
        if bottom_portal_y < ball_y_latest < top_portal_y:
            if (ood_latest_step['ball_y'] - prev_time_step['ball_y']) > 0:
                ood_latest_step['ball_y'] = top_portal_y
            else:
                ood_latest_step['ball_y'] = bottom_portal_y
        # endregion
        # Noise addition
        for factor, latest in (('ball_x', ball_x_latest), ('ball_y', ball_y_latest), ('ball_vel_dir', ball_vel_dir_latest)):
            noise = np.random.randn()
            id_new_time_step[factor] = latest + noise * settings[f'{factor}_noise']
            ood_latest_step[factor] += noise * settings[f'{factor}_noise']


    incr_score_left, incr_score_right = prev_time_step['score_left'] + int(point_left), prev_time_step['score_right'] + int(point_right)
    id_new_time_step['score_left'], id_new_time_step['score_right'] = incr_score_left, incr_score_right
    if max(incr_score_left, incr_score_right) >= settings['max_points']:
        reset_score_left, reset_score_right = 0, 0
        id_new_time_step['score_left'], id_new_time_step['score_right'] = reset_score_left, reset_score_right

    #region score OOD change: if there was a goal (ball_reset) the OOD score is reset to 1 instead of 0
    ood_latest_step['score_left'], ood_latest_step['score_right'] = incr_score_left, incr_score_right
    if max(incr_score_left, incr_score_right) >= settings['max_points']:
        reset_score_left, reset_score_right = 1, 1
        ood_latest_step['score_left'], ood_latest_step['score_right'] = reset_score_left, reset_score_right
    #endregion

    id_new_time_step['ball_vel_magn'] = ball_vel_magn_latest  # Constant velocity magnitude
    # region ball_vel_magn OOD change: set velocity magnitude to 4.0 instead of 2.0 when the ball was in the bottom half of the screen
    ood_latest_step['ball_vel_magn'] = ball_vel_magn_latest
    if prev_time_step['ball_y'] < settings['center_point_y']:
        ood_latest_step['ball_vel_magn'] = 4.0
    else:
        ood_latest_step['ball_vel_magn'] = 2.0
    # ood_latest_step['ball_vel_magn'] = 2.0 + .2 * min(prev_time_step['score_left'], prev_time_step['score_right'])
    # endregion

    merged_new_time_step = {k:id_new_time_step[k] if (k not in settings['ood']) else ood_latest_step[k] for k in id_new_time_step}
    # id_new_time_step = put_in_boundaries(id_new_time_step, settings)
    # return id_new_time_step, ball_reset
    merged_new_time_step = put_in_boundaries(merged_new_time_step, settings)
    return merged_new_time_step, ball_reset

def check_point(natural_next_bx, settings):
    point_left, point_right = False, False
    if natural_next_bx < settings['ball_x_min_point']:
        point_right = True
    elif natural_next_bx > settings['ball_x_max_point']:
        point_left = True
    return point_left, point_right

def next_step(prev_time_step, settings, args):
    """
    Combines the observational and interventional dynamics to sample a new step with respecting the intervention sample policy.
    """
    nsr_function = alt_next_step_regular if not args.og_next_step_regular else next_step_regular
    std_time_step, ball_reset = nsr_function(prev_time_step, settings) # 8 factors: 'paddle_left_y', 'paddle_right_y', 'ball_x', 'ball_y', 'ball_vel_dir', 'score_left', 'score_right', 'ball_vel_magn
    intv_time_step = next_step_interventions(prev_time_step, settings, ball_reset=ball_reset) # 5 intervenable factors: 'paddle_left_y', 'paddle_right_y', 'ball_x', 'ball_y', 'ball_vel_dir'. 3 non-intervenable factors: 'score_left', 'score_right', 'ball_vel_magn'
    intv_targets = {}
    if settings['single_target_interventions']:
        keys = sorted(list(intv_time_step.keys()))
        num_vars = len(keys)
        t = np.random.randint(num_vars)
        no_int_prob = (1 - settings['intv_prob']) ** num_vars
        if np.random.uniform() < no_int_prob:
            t = -1
        intv_targets = {key: int(t == i) for i, key in enumerate(keys)}
    else:
        intv_targets = {key: int(np.random.rand() < settings['intv_prob']) for key in intv_time_step}
    for key in std_time_step:
        if key in intv_time_step and intv_targets[key] == 1:
            std_time_step[key] = intv_time_step[key]
        else:
            intv_targets[key] = 0
    return std_time_step, intv_targets


def create_settings(args=None):
    """
    Create a dictionary of the general hyperparameters for generating the Pong dataset.
    """
    border_size = 2
    settings = {
        'resolution': 32,
        'dpi': 1,
        'border_size': border_size,
        'paddle_height': 6,
        'paddle_width': 2,
        'paddle_left_x': 3 + border_size,
        'paddle_left_y_min': border_size,
        'paddle_right_y_min': border_size,
        'paddle_max_step': 1.5, # in paper, mean step size .05 (but times 30, so 1.5)
        'paddle_step_noise': 0.5, # in paper: .017 (but times 30, so .5)
        'ball_radius': 1.2,
        'ball_x_noise': 0.2, # .015 in the paper, but .2 in the zenodo settings.json. .015*30=.45, so this is different
        'ball_y_noise': 0.2, # .015 in the paper, but .2 in the zenodo settings.json .015*30=.45, so this is different
        'ball_vel_dir_noise': 0.1, # .015 in the paper, but .1 in the zenodo settings.json  .015*30=.45, so this is different
        'max_points': 5,
        'intv_prob': 0.15, # .65 in the paper, but .15 in the zenodo settings.json
        'single_target_interventions': True
        # Zenodo file has 250000 points, paper mentions 100 000 points
        # The zenodo settings.json also includes the following, which are not compatible with this code:
        # 'ball_vel_magn_min': 1.0,
        # 'ball_vel_magn_max': 3.0,
        # 'ball_vel_magn_noise': 0.05,
    }
    if args.paper_settings:
        paper_settings = { # See https://arxiv.org/pdf/2202.03169.pdf#page=29
            # 'paddle_mean_step': 0.05,
            'paddle_max_step': 0.05,
            'paddle_step_noise': 0.017,
            'ball_x_noise': 0.015,
            'ball_y_noise': 0.015,
            'ball_vel_dir_noise': 0.015,
            # 'intv_prob': 0.65
            'intv_prob': 0.19 # See https://github.com/phlippe/CITRIS/issues/7
        }
        for key in paper_settings:
            settings[key] = paper_settings[key]

    if args.paddle_max_step is not None:
        settings['paddle_max_step'] = args.paddle_max_step
    if args.intv_prob is not None:
        settings['intv_prob'] = args.intv_prob

    settings['paddle_right_x'] = settings['resolution'] - settings['paddle_left_x']
    settings['paddle_left_y_max'] = settings['resolution'] - settings['border_size']
    settings['paddle_right_y_max'] = settings['resolution'] - settings['border_size']
    settings['ball_y_max'] = settings['resolution'] - settings['ball_radius'] - settings['border_size']
    settings['ball_y_min'] = settings['ball_radius'] + settings['border_size']
    settings['ball_x_max_point'] = settings['resolution'] - settings['ball_radius'] - border_size
    settings['ball_x_min_point'] = settings['ball_radius'] + border_size
    settings['ball_x_max'] = settings['resolution'] - border_size
    settings['ball_x_min'] = border_size
    settings['ball_x_max_sample'] = settings['paddle_right_x'] - settings['paddle_width'] / 2.0 - settings['ball_radius']
    settings['ball_x_min_sample'] = settings['paddle_left_x'] + settings['paddle_width'] / 2.0 + settings['ball_radius']
    settings['center_point_x'] = settings['resolution'] / 2.0
    settings['center_point_y'] = settings['resolution'] / 2.0
    return settings


def sample_random_point(settings):
    """
    Sample a completely independent, random point for all causal factors.
    """
    step = dict()
    step['ball_x'] = np.random.uniform(settings['ball_x_min_sample'], settings['ball_x_max_sample'])
    step['ball_y'] = np.random.uniform(settings['ball_y_min'], settings['ball_y_max'])
    step['ball_vel_magn'] = 2.0
    step['ball_vel_dir'] = sample_ball_vel_dir(settings)
    step['paddle_left_y'] = np.random.uniform(settings['paddle_left_y_min'], settings['paddle_left_y_max'])
    step['paddle_right_y'] = np.random.uniform(settings['paddle_right_y_min'], settings['paddle_right_y_max'])
    step['score_left'] = np.random.randint(settings['max_points'])
    step['score_right'] = np.random.randint(settings['max_points'])
    return step


def plot_matplotlib_figure(time_step, settings, filename):
    """
    Render a time step with matplotlib
    """
    plt.figure(figsize=(settings['resolution'], settings['resolution']), dpi=settings['dpi'])
    ax = plt.gca()
    ax.set_aspect('equal', 'box')
    plt.xlim(0, settings['resolution'])
    plt.ylim(0, settings['resolution'])
    plt.yticks([])
    plt.xticks([])
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)

    color_back = np.array([0.9, 0.9, 0.9])
    background = plt.Rectangle((0, 0), settings['resolution'], settings['resolution'], fc=color_back)
    ax.add_patch(background)
    digit_color = np.array([0.5, 0.5, 0.5])
    write_digit(x=settings['resolution']/2.0-4, y=24, digit=time_step['score_left'], color=digit_color, ax=ax)
    rec = plt.Rectangle((settings['resolution']/2.0, 25), 1, 1, fc=digit_color)
    ax.add_patch(rec)
    rec = plt.Rectangle((settings['resolution']/2.0, 27), 1, 1, fc=digit_color)
    ax.add_patch(rec)
    write_digit(x=settings['resolution']/2.0+2, y=24, digit=time_step['score_right'], color=digit_color, ax=ax)
    paddle_left = plt.Rectangle((settings['paddle_left_x'] - settings['paddle_width']/2.0, time_step['paddle_left_y'] - settings['paddle_height']/2.0), settings['paddle_width'], settings['paddle_height'], fc='b', snap=False)
    ax.add_patch(paddle_left)
    paddle_right = plt.Rectangle((settings['paddle_right_x'] - settings['paddle_width']/2.0, time_step['paddle_right_y'] - settings['paddle_height']/2.0), settings['paddle_width'], settings['paddle_height'], fc='g', snap=False)
    ax.add_patch(paddle_right)
    border_left = plt.Rectangle((0, 0), settings['border_size'], settings['resolution'], fc=np.array([0.1,0.1,0.1]))
    ax.add_patch(border_left)
    border_right = plt.Rectangle((settings['resolution']-settings['border_size'], 0), settings['border_size'], settings['resolution'], fc=np.array([0.1,0.1,0.1]))
    ax.add_patch(border_right)
    border_bottom = plt.Rectangle((0, 0), settings['resolution'], settings['border_size'], fc=np.array([0.1,0.1,0.1]))
    ax.add_patch(border_bottom)
    border_top = plt.Rectangle((0, settings['resolution']-settings['border_size']), settings['resolution'], settings['border_size'], fc=np.array([0.1,0.1,0.1]))
    ax.add_patch(border_top)
    ball = plt.Circle((time_step['ball_x'], time_step['ball_y']), radius=settings['ball_radius'], fc='r')
    ax.add_patch(ball)
    
    plt.savefig(filename)
    plt.close()

    plt.figure(figsize=(settings['resolution'], settings['resolution']), dpi=settings['dpi'])
    ax = plt.gca()
    ax.set_aspect('equal', 'box')
    plt.xlim(0, settings['resolution'])
    plt.ylim(0, settings['resolution'])
    plt.yticks([])
    plt.xticks([])
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)

    vel_x = np.sin(time_step['ball_vel_dir']) * time_step['ball_vel_magn']
    vel_y = np.cos(time_step['ball_vel_dir']) * time_step['ball_vel_magn']
    ball_x_proj = time_step['ball_x'] + vel_x
    ball_y_proj = time_step['ball_y'] + vel_y
    ball = plt.Circle((ball_x_proj, ball_y_proj), radius=settings['ball_radius'], fc=np.array([0.0,0.0,0.0]))
    ax.add_patch(ball)

    plt.savefig(filename.replace('.png', '_proj.png'))
    plt.close()


def write_digit(x, y, digit, color, ax):
    """
    Writing the score digits as matplotlib rectangles
    """
    if digit == 0:
        rec = plt.Rectangle((x, y), 3, 1, fc=color)
        ax.add_patch(rec)
        rec = plt.Rectangle((x, y), 1, 5, fc=color)
        ax.add_patch(rec)
        rec = plt.Rectangle((x+2, y), 1, 5, fc=color)
        ax.add_patch(rec)
        rec = plt.Rectangle((x, y+4), 3, 1, fc=color)
        ax.add_patch(rec)
    if digit == 1:
        rec = plt.Rectangle((x+1, y), 1, 5, fc=color)
        ax.add_patch(rec)
    if digit == 2:
        rec = plt.Rectangle((x, y), 3, 1, fc=color)
        ax.add_patch(rec)
        rec = plt.Rectangle((x, y+2), 3, 1, fc=color)
        ax.add_patch(rec)
        rec = plt.Rectangle((x, y+4), 3, 1, fc=color)
        ax.add_patch(rec)
        rec = plt.Rectangle((x+2, y+2), 1, 3, fc=color)
        ax.add_patch(rec)
        rec = plt.Rectangle((x, y), 1, 3, fc=color)
        ax.add_patch(rec)
    if digit == 3:
        rec = plt.Rectangle((x+2, y), 1, 5, fc=color)
        ax.add_patch(rec)
        rec = plt.Rectangle((x, y), 3, 1, fc=color)
        ax.add_patch(rec)
        rec = plt.Rectangle((x, y+2), 3, 1, fc=color)
        ax.add_patch(rec)
        rec = plt.Rectangle((x, y+4), 3, 1, fc=color)
        ax.add_patch(rec)
    if digit == 4:
        rec = plt.Rectangle((x+2, y), 1, 5, fc=color)
        ax.add_patch(rec)
        rec = plt.Rectangle((x, y+2), 3, 1, fc=color)
        ax.add_patch(rec)
        rec = plt.Rectangle((x, y+2), 1, 3, fc=color)
        ax.add_patch(rec)
    if digit == 5:
        rec = plt.Rectangle((x, y), 3, 1, fc=color)
        ax.add_patch(rec)
        rec = plt.Rectangle((x, y+2), 3, 1, fc=color)
        ax.add_patch(rec)
        rec = plt.Rectangle((x, y+4), 3, 1, fc=color)
        ax.add_patch(rec)
        rec = plt.Rectangle((x+2, y), 1, 3, fc=color)
        ax.add_patch(rec)
        rec = plt.Rectangle((x, y+2), 1, 3, fc=color)
        ax.add_patch(rec)


def create_seq_dataset(num_samples, folder,args):
    """
    Generate a dataset consisting of a single sequence with num_samples data points.
    Does not include the rendering with matplotlib.
    """

    os.makedirs(folder, exist_ok=True)

    settings = create_settings(args)
    settings['ood'] = args.ood
    if args.no_noise:
        settings['paddle_step_noise'] = 0.0
        settings['ball_x_noise'] = 0.0
        settings['ball_y_noise'] = 0.0
        settings['ball_vel_dir_noise'] = 0.0
    if args.no_intvs:
        settings['intv_prob'] = 0.0
    start_point = sample_random_point(settings)
    start_point, _ = next_step(start_point, settings, args)
    key_list = sorted(list(start_point.keys()))
    all_steps = np.zeros((num_samples, len(key_list)), dtype=np.float32)
    all_interventions = np.zeros((num_samples-1, len(key_list)), dtype=np.float32)
    next_time_step = start_point
    for n in range(num_samples):
        next_time_step, intv_targets = next_step(next_time_step, settings, args)
        for i, key in enumerate(key_list):
            all_steps[n, i] = next_time_step[key]
            if n > 0:
                all_interventions[n-1, i] = intv_targets[key]

    np.savez_compressed(os.path.join(folder, 'latents.npz'), 
                        latents=all_steps, 
                        targets=all_interventions, 
                        keys=key_list)
    with open(os.path.join(folder, 'settings.json'), 'w') as f:
        json.dump(settings, f, indent=4)


def create_indep_dataset(num_samples, folder, args=None):
    """
    Generate a dataset with independent samples. Used for correlation checking.
    """
    os.makedirs(folder, exist_ok=True)

    settings = create_settings(args)
    start_point = sample_random_point(settings)
    key_list = sorted(list(start_point.keys()))
    all_steps = np.zeros((num_samples, len(key_list)), dtype=np.float32)
    for n in range(num_samples):
        new_step = sample_random_point(settings)
        for i, key in enumerate(key_list):
            all_steps[n, i] = new_step[key]

    np.savez_compressed(os.path.join(folder, 'latents.npz'), 
                        latents=all_steps,
                        targets=np.ones_like(all_steps),
                        keys=key_list)
    with open(os.path.join(folder, 'settings.json'), 'w') as f:
        json.dump(settings, f, indent=4)


def create_triplet_dataset(num_samples, folder, start_data):
    """
    Generate a dataset for the triplet evaluation.
    """
    os.makedirs(folder, exist_ok=True)
    start_dataset = np.load(start_data)
    images = start_dataset['images']
    latents = start_dataset['latents']
    targets = start_dataset['targets']
    has_intv = (targets.sum(axis=0) > 0).astype(np.int32)

    all_latents = np.zeros((num_samples, 3, latents.shape[1]), dtype=np.float32)
    prev_images = np.zeros((num_samples, 2) + images.shape[1:], dtype=np.uint8)
    target_masks = np.zeros((num_samples, latents.shape[1]), dtype=np.uint8)
    for n in range(num_samples):
        # Pick two random images that we want to combine
        idx1 = np.random.randint(images.shape[0])
        idx2 = np.random.randint(images.shape[0]-1)
        if idx2 >= idx1:
            idx2 += 1
        latent1 = latents[idx1]
        latent2 = latents[idx2]
        # Pick a random combination of both images for a new, third image
        srcs = None if has_intv.sum() > 0 else np.random.randint(2, size=(latent1.shape[0],))
        while srcs is None or srcs.astype(np.float32).std() == 0.0:  
            # Prevent that we take all causal variables from one of the two images
            srcs = np.random.randint(2, size=(latent1.shape[0],))
            srcs = srcs * has_intv if has_intv.sum() > 0 else srcs
        latent3 = np.where(srcs == 0, latent1, latent2)
        all_latents[n,0] = latent1
        all_latents[n,1] = latent2
        all_latents[n,2] = latent3
        prev_images[n,0] = images[idx1]
        prev_images[n,1] = images[idx2]
        target_masks[n] = srcs

    np.savez_compressed(os.path.join(folder, 'latents.npz'), 
                        latents=all_latents[:,-1],
                        triplet_latents=all_latents,
                        triplet_images=prev_images,
                        triplet_targets=target_masks,
                        keys=start_dataset['keys'])


def export_figures(folder, start_index=0, end_index=-1):
    """
    Given a numpy array of latent variables, render each data point with matplotlib.
    """
    if isinstance(folder, tuple):
        folder, start_index, end_index = folder
    latents_arr = np.load(os.path.join(folder, 'latents.npz'))
    latents = latents_arr['latents']
    keys = latents_arr['keys'].tolist()
    
    with open(os.path.join(folder, 'settings.json'), 'r') as f:
        settings = json.load(f)

    if end_index < 0:
        end_index = latents.shape[0]

    figures = np.zeros((end_index - start_index, settings['resolution']*settings['dpi'], settings['resolution']*settings['dpi'], 4), dtype=np.uint8)
    t = time.time()
    for i in range(start_index, end_index):
        time_step = {key: latents[i,j] for j, key in enumerate(keys)}
        filename = os.path.join(folder, f'fig_{str(start_index).zfill(7)}.png')
        plot_matplotlib_figure(time_step=time_step, 
                               settings=settings, 
                               filename=filename)
        main_img = imread(filename)[:,:,:3]
        move_img = imread(filename.replace('.png', '_proj.png'))[:,:,:1]
        figures[i - start_index,:,:,:3] = main_img
        figures[i - start_index,:,:,3:] = move_img
        if time.time() - t > 10:
            print(f'Processed {i-start_index} images out of {end_index - start_index}')
            t = time.time()

    if start_index != 0 or end_index < latents.shape[0]:
        output_filename = os.path.join(folder, f'images_{str(start_index).zfill(8)}_{str(end_index).zfill(8)}.npz')
    else:
        output_filename = os.path.join(folder, 'images.npz')
    np.savez_compressed(output_filename,
                        imgs=figures)


def generate_full_dataset(dataset_size, folder, args, split_name=None, num_processes=8, independent=False, triplets=False, start_data=None):
    """
    Generate a full dataset from latent variables to rendering with matplotlib.
    To speed up the rendering process, we parallelize it with using multiple processes.
    """

    if independent:
        create_indep_dataset(dataset_size, folder, args)
    elif triplets:
        create_triplet_dataset(dataset_size, folder, start_data=start_data)
    else:
        create_seq_dataset(dataset_size, folder,args)

    print(f'Starting figure export... with {num_processes} process{"es" if num_processes != 1 else ""}')
    start_time = time.time()
    if num_processes > 1:
        inp_args = []
        for i in range(num_processes):
            start_index = dataset_size//num_processes*i
            end_index = dataset_size//num_processes*(i+1)
            if i == num_processes - 1:
                end_index = -1
            inp_args.append((folder, start_index, end_index))
        with Pool(num_processes) as p:
            p.map(export_figures, inp_args)
        # Merge datasets
        img_sets = sorted(glob(os.path.join(folder, 'images_*_*.npz')))
        images = np.concatenate([np.load(s)['imgs'] for s in img_sets], axis=0)
        np.savez_compressed(os.path.join(folder, 'images.npz'),
                            imgs=images)
        for s in img_sets:
            os.remove(s)
        extra_imgs = sorted(glob(os.path.join(folder, 'fig_*.png')))
        for s in extra_imgs:
            os.remove(s)
    else:
        export_figures(folder)
    print(f'Finished figure export in {time.time() - start_time:.2f} seconds')
    if split_name is not None:
        images = np.load(os.path.join(folder, 'images.npz'))
        latents = np.load(os.path.join(folder, 'latents.npz'))
        if triplets:
            elem_dict = dict()
            elem_dict['latents'] = latents['triplet_latents']
            elem_dict['targets'] = latents['triplet_targets']
            elem_dict['keys'] = latents['keys']
            elem_dict['images'] = np.concatenate([latents['triplet_images'], images['imgs'][:,None]], axis=1)
        else:
            elem_dict = {key: latents[key] for key in latents.keys()}
            elem_dict['images'] = images['imgs']
        np.savez_compressed(os.path.join(folder, split_name + '.npz'),
                            **elem_dict)
        os.remove(os.path.join(folder, 'images.npz'))
        os.remove(os.path.join(folder, 'latents.npz'))

    end_time = time.time()
    dur = int(end_time - start_time)
    print(f'Finished in {dur // 60}min {dur % 60}s')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--output_folder', type=str, required=True,
                        help='Folder to save the dataset to.')
    parser.add_argument('--dataset_size', type=int, default=100000,
                        help='Number of samples to use for the dataset.')
    parser.add_argument('--num_processes', type=int, default=8,
                        help='Number of processes to use for the rendering.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for reproducibility.')
    parser.add_argument('--ood', type=str, nargs='+',
                        default=None)  # Nathan: which factor mechanism(s) to replace by an ood mechanism
    parser.add_argument('--no_noise', action='store_true')
    parser.add_argument('--no_intvs', action='store_true')
    parser.add_argument('--splits', nargs='*', default=['train', 'val', 'val_indep', 'val_triplets', 'test', 'test_indep', 'test_triplets'])
    parser.add_argument('--paper_settings', action='store_true')
    parser.add_argument('--paddle_max_step', default=None, type=float)
    parser.add_argument('--intv_prob', default=None, type=float)
    parser.add_argument('--og_next_step_regular', action='store_true')
    args = parser.parse_args()
    if args.ood is None:
        args.ood = []
    args.output_folder = os.path.join(args.output_folder, f"{args.dataset_size}_train_points{'_no_intvs' if args.no_intvs else ''}{'_no_noise' if args.no_noise else ''}{'_paper_settings' if args.paper_settings else ''}","__".join(args.ood))
    np.random.seed(args.seed)

    if 'train' in args.splits:
        generate_full_dataset(args.dataset_size,
                              folder=args.output_folder,
                              split_name='train', args=args,
                              num_processes=args.num_processes)
    if 'val' in args.splits:
        generate_full_dataset(args.dataset_size // 10,
                              folder=args.output_folder,
                              split_name='val', args=args,
                              num_processes=args.num_processes)
    if 'val_indep' in args.splits:
        generate_full_dataset(args.dataset_size // 4,
                              folder=args.output_folder,
                              split_name='val_indep', args=args,
                              independent=True,
                              num_processes=args.num_processes)
    if 'val_triplets' in args.splits:
        generate_full_dataset(args.dataset_size // 10,
                              folder=args.output_folder,
                              split_name='val_triplets', args=args,
                              triplets=True,
                              start_data=os.path.join(args.output_folder, 'val.npz'),
                              num_processes=args.num_processes)
    if 'test' in args.splits:
        generate_full_dataset(args.dataset_size // 10,
                              folder=args.output_folder,
                              split_name='test', args=args,
                              num_processes=args.num_processes)
    if 'test_indep' in args.splits:
        generate_full_dataset(args.dataset_size // 4,
                              folder=args.output_folder,
                              split_name='test_indep',
                              independent=True, args=args,
                              num_processes=args.num_processes)
    if 'test_triplets' in args.splits:
        generate_full_dataset(args.dataset_size // 10,
                              folder=args.output_folder,
                              split_name='test_triplets',
                              triplets=True, args=args,
                              start_data=os.path.join(args.output_folder, 'test.npz'),
                              num_processes=args.num_processes)
    # Nathan
    if 'train_triplets' in args.splits:
        generate_full_dataset(args.dataset_size,
                              folder=args.output_folder,
                              split_name='train_triplets', args=args,
                              triplets=True,
                              start_data=os.path.join(args.output_folder, 'train.npz'),
                              num_processes=args.num_processes)