#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __author__=c"Frank Jing"


import copy
import random

import numpy as np
import pandas as pd

from typing import *


def ends_to_dir_vec(ends: list):
    """color ends presentation to direction vector presentation. ends: [z_ends, y_ends, z_ends]"""
    dir_vec = {}
    for i_ax, ax in enumerate(ends):
        for i_end, color in enumerate(ax):
            if color == '-':
                continue
            vec = [0, 0, 0]
            vec[i_ax] = -1 if i_end == 0 else 1
            dir_vec[color] = np.array(vec, dtype=int)
    return dir_vec


def dir_vec_to_ends(vec_dic: dict):
    ends = [['-', '-'], ['-', '-'], ['-', '-']]
    for color, vector in vec_dic.items():
        i_axis = np.argwhere(vector != 0).squeeze().tolist()
        i_end = 0 if vector.sum() < 0 else 1
        ends[i_axis][i_end] = color
    return ends


MAGIC_ORDER = 3


r, o = 'r', 'o'  # red and orange
y, w = 'y', 'w'  # yellow and white
b, g = 'b', 'g'  # blue and green
# magic1 cube gestures
# note: axis order and direction must apply right-hand law: e0 X e1 = e2; e1 X e2 = e3; e2 X e0 = e1
MAIN_X = (r, o)  # x(2nd) axis, from left to right
MAIN_Y = (y, w)  # y(1st) axis, from up to down
MAIN_Z = (b, g)  # z(0th) axis, from outer to inner

MAIN_GESTURE = ends_to_dir_vec([MAIN_Z, MAIN_Y, MAIN_X])


class Cell(object):
    def __init__(self, x: list=None, y: list=None, z: list=None):
        x = x if x is not None else ['-', '-']
        y = y if y is not None else ['-', '-']
        z = z if z is not None else ['-', '-']
        self.gesture = ends_to_dir_vec([z, y, x])

    def rot90_right_hand(self, rot_vec: np.ndarray, k):
        """rotate 90 degrees along vector(determined by stator_axis and is_ind_dir) by right hand law"""
        for color, dir_vec in self.gesture.items():
            if np.dot(rot_vec, dir_vec) == 0:  # if two vectors are orthogonal
                for _ in range(k):
                    self.gesture[color] = np.cross(rot_vec, self.gesture[color])

    def get_color_by_dir(self, dir):
        for color, direction in self.gesture.items():
            if (dir == direction).all():
                return color
        print(self.gesture)
        raise Exception('Not Found!')

    def __repr__(self):
        ends = dir_vec_to_ends(self.gesture)
        x = f'[{ends[2][0]},{ends[2][1]}]'
        y = f'[{ends[1][0]},{ends[1][1]}]'
        z = f'[{ends[0][0]},{ends[0][1]}]'
        return f'[{z},{y},{x}]'

    def __str__(self):
        return self.__repr__()


def main_color_to_axis_and_position(color: str):
    vec = MAIN_GESTURE[color]
    i_axis = np.argwhere(vec != 0).squeeze().tolist()
    i_end = 0 if vec.sum() < 0 else 1
    return i_axis, (MAGIC_ORDER - 1) * i_end


def make_magic(*six_plain_color_seq: str):
    """every plain color must be sequenced from left to right then up to down.
    two char before ':' used for locator, means: down-color and right-color """
    cells = list(map(lambda u: Cell(), range(MAGIC_ORDER**3)))
    magic = np.array(cells).reshape([MAGIC_ORDER, ]*3)
    # every plain of magic1 cube
    for color_seq in six_plain_color_seq:
        down_right_color, plain_seq = color_seq.split(':')
        down_color, right_color = down_right_color
        down_axis, _ = main_color_to_axis_and_position(down_color)
        right_axis, _ = main_color_to_axis_and_position(right_color)
        inner_color = plain_seq[len(plain_seq)//2]
        # plain color array, from left to right then from up to down.
        plain_array = np.array(list(map(lambda u: list(u), plain_seq.split())))
        if MAIN_GESTURE[down_color].sum() < 0:
            plain_array = np.flipud(plain_array)
        if MAIN_GESTURE[right_color].sum() < 0:
            plain_array = np.fliplr(plain_array)
        if down_axis > right_axis:
            plain_array = plain_array.transpose()
        i_axis, i_pos = main_color_to_axis_and_position(inner_color)
        # print('='*30 + color_seq)
        # print(plain_array)
        # every row of plain color sequence
        for i_row in range(MAGIC_ORDER):
            # every color(cell surface) of row
            for i_col in range(MAGIC_ORDER):
                idx = [i_row, i_col]
                idx.insert(i_axis, i_pos)
                # set color orientation.
                color = plain_array[i_row, i_col]
                if color in magic[idx[0], idx[1], idx[2]].gesture.keys():
                    print(idx)
                    print(color, magic[idx[0], idx[1], idx[2]].gesture)
                # print(f'idx: {idx}, color: {color}, dir: {MAIN_GESTURE[inner_color]}')
                magic[idx[0], idx[1], idx[2]].gesture[color] = MAIN_GESTURE[inner_color]

    return magic


def magic_rot90_right_hand(magic, main_color, k):
    """rotate 90 degrees along vector(determined by stator_axis and is_ind_dir) by right hand law"""
    magic2 = copy.deepcopy(magic)
    stator_axis, i_pos = main_color_to_axis_and_position(main_color)
    is_inc_dir = i_pos > 0
    rotor_axes = [0, 1, 2]
    rotor_axes.pop(stator_axis)
    # if stator_axis==1(y axis), should reverse one time, if decrease direction, should reverse one time.
    reverse = (stator_axis == 1) ^ (not is_inc_dir)
    if reverse:
        rotor_axes.reverse()
    idx = [np.s_[:], np.s_[:]]
    idx.insert(stator_axis, np.s_[i_pos:i_pos + 1])
    magic2[tuple(idx)] = np.rot90(magic2[tuple(idx)], k=k, axes=rotor_axes)
    # rotate magic1 cell
    for i_row in range(MAGIC_ORDER):
        for i_col in range(MAGIC_ORDER):
            idx = [i_row, i_col]
            idx.insert(stator_axis, i_pos)
            magic2[idx[0], idx[1], idx[2]].rot90_right_hand(MAIN_GESTURE[main_color], k)

    return magic2


DOT_DIST_TABLE = pd.DataFrame([
    [0, 1, 1],
    [1, None, 2],
    [1, 2, 2]
], index=[+1, 0, -1], columns=[+1, 0, -1])


def cell_distance(cell):
    if len(cell.gesture) < 2:
        return 0
    colors = cell.gesture.keys()
    c_dirs = list(map(lambda c: cell.gesture[c], colors))
    m_dirs = list(map(lambda c: MAIN_GESTURE[c], colors))
    dots = list(map(lambda tup: np.dot(tup[0], tup[1]), zip(m_dirs, c_dirs)))
    if len(cell.gesture) == 2:
        dist = DOT_DIST_TABLE.loc[dots[0], dots[1]]
        if pd.isna(dist):
            rot0, rot1 = np.cross(c_dirs[0], m_dirs[0]), np.cross(c_dirs[1], m_dirs[1])
            unavailable_rot = np.cross(c_dirs[0], c_dirs[1])
            # rot1, rot2 are both unavailable
            if np.dot(rot0, unavailable_rot) != 0 and np.dot(rot1, unavailable_rot) != 0:
                dist = 3
            else:
                dist = 2
    else:  # len(cell.gesture) == 3:
        n_same_dir = (np.array(dots) == 1).sum()
        if n_same_dir == 0:
            dist = 2
        elif n_same_dir == 1:
            dist = 1
        else:
            dist = 0
    return dist


def magic_distance(magic):
    return np.array(list(map(cell_distance, magic.flatten().tolist())), dtype=int).sum()


def magic_score(magic):
    return np.array(list(map(lambda m: cell_distance(m) <= 1, magic.flatten().tolist())), dtype=int).sum()

def print_magic(magic):

    def sprint_plain(inner_color, down_color, right_color):
        plain_array = np.array(['-', ]*MAGIC_ORDER**2,).reshape([MAGIC_ORDER, MAGIC_ORDER])
        i_axis, i_pos = main_color_to_axis_and_position(inner_color)
        for i_row in range(MAGIC_ORDER):
            for i_col in range(MAGIC_ORDER):
                idx = [i_row, i_col]
                idx.insert(i_axis, i_pos)
                plain_array[i_row, i_col] = magic[idx[0], idx[1], idx[2]].get_color_by_dir(MAIN_GESTURE[inner_color])
        # plain color array, from left to right then from up to down.
        down_axis, _ = main_color_to_axis_and_position(down_color)
        right_axis, _ = main_color_to_axis_and_position(right_color)
        if down_axis > right_axis:
            plain_array = plain_array.transpose()
        if MAIN_GESTURE[down_color].sum() < 0:
            plain_array = np.flipud(plain_array)
        if MAIN_GESTURE[right_color].sum() < 0:
            plain_array = np.fliplr(plain_array)
        plain_seq = f'{down_color}{right_color}:'
        for row in plain_array:
            row_str = ''.join(row.tolist())
            plain_seq = f'{plain_seq}{row_str} '

        return plain_seq[:-1]

    print(sprint_plain(y, g, o))
    print(sprint_plain(g, w, o))
    print(sprint_plain(o, w, b))
    print(sprint_plain(b, w, r))
    print(sprint_plain(r, w, g))
    print(sprint_plain(w, b, o))


MAGIC = make_magic(
    'go:yyy gyo oyr',
    'wo:grg ggg ggg',
    'wb:yyo ooo ooo',
    'wr:bbb bbb bbb',
    'wg:ryy rrr rrr',
    'bo:www www www'
)
ops = []
magic1 = copy.deepcopy(MAGIC)

BASE_OPS = [(main_color, k) for main_color in MAIN_GESTURE.keys() for k in [1, 2, 3]]

# print_magic(MAGIC)
# for _ in range(5):
#     m_color, k = BASE_OPS[random.randint(0, len(BASE_OPS) - 1)]
#     # m_color, k = 'g', 1
#     print('='*20 + f'main color: {m_color}, k: {k}')
#     magic1 = magic_rot90_right_hand(magic1, m_color, k)
#     print_magic(magic1)


# while True:
#     score1 = magic_score(magic1)
#     res = np.array(list(map(lambda op: magic_score(magic_rot90_right_hand(magic1, op[0], op[1])), BASE_OPS)))
#     m_color, k = BASE_OPS[res.argmax()]
#     magic1 = magic_rot90_right_hand(magic1, m_color, k)
#     score2 = magic_score(magic1)
#     print(f'm_color: {m_color}, k: {k}, score1: {score1}, score2: {score2}')
#     if score2 <= score1:
#         c, k = random.choice(BASE_OPS)
#         magic1 = magic_rot90_right_hand(magic1, c, k)
#         # c, k = random.choice(BASE_OPS)
#         # magic1 = magic_rot90_right_hand(magic1, c, k)

SOLVED = False
def solve(magic, ops: list):
    global  SOLVED
    # print(ops)
    if SOLVED or len(ops) > 5:
        return
    if len(ops) > 0:
        last_m_color, _ = ops[-1]
        base_ops2 = list(filter(lambda u: u[0] != last_m_color, BASE_OPS))
    else:
        base_ops2 = BASE_OPS
    random.shuffle(base_ops2)
    for m_color, k in base_ops2:
        ops.append((m_color, k))
        magic2 = magic_rot90_right_hand(magic, m_color, k)
        if magic_distance(magic2) > 0:
            solve(magic2, ops)
            ops.pop(-1)
        else:
            SOLVED = True


solve_ops = []
solve(magic1, solve_ops)



# while True:
#     score1 = magic_score(magic1)
#     print('*'*20 + f'score1: {score1}, id: {id(magic1)}')
#     base_ops2 = BASE_OPS.copy()
#     random.shuffle(base_ops2)
#     for main_color, k in base_ops2:
#         magic2 = magic_rot90_right_hand(magic1, main_color, k)
#         score2 = magic_score(magic2)
#         if score2 > score1:
#             magic1 = magic2
#             ops.append((main_color, k))
#             print(f'score1: {score1}, score2: {score2}, op: ({main_color}, {k})')
#             break









