# Source:
# https://github.com/tomrunia/OpticalFlow_Visualization/blob/master/flow_vis/flow_vis.py
# A few modifications made 2023-07-01, by Avi Ziskind, including adding options to
#  - make zero flow slightly gray instead of white. 
#  - add an alpha channel to the output (for easy matplotlib plotting)
#  - convert to floating point (0..1) instead of uint8 (0..255).

# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-08-03

import numpy as np


def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
      Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
      URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col = col + RY
    # YG
    colorwheel[col:col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col:col + YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col = col + GC
    # CB
    colorwheel[col:col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col:col + CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col = col + BM
    # MR
    colorwheel[col:col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col:col + MR, 0] = 255
    return colorwheel


def flow_uv_to_colors_raw(u, v, convert_to_bgr=False, zero_gray_value=1.0, alpha=None,
                          to_float=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
        zero_gray_value (bool, optional): grayscale color of zero-flow.
            Default is 1 (white)
        alpha (float, optional): if provided, add an alpha channel populated
            with this value
        to_float (bool, optional): indicate whether to divide by 255 and return floating 
            point numbers (default False, which returns uint8 values from 0 to 255)

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1
        idx = (rad <= 1)
        zg = zero_gray_value

        # original code. rad=0 => col=1;  rad=1 => col=col
        # col[idx] = 1 - rad[idx] * (1 - col[idx])

        # new code: rad=0 => col=zero_gray_value;  rad=1 => col=col
        col[idx] = zg - zg * rad[idx] * (1 - col[idx]/zg)

        col[~idx] = col[~idx] * 0.75  # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2 - i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * col)

    # add alpha channel:
    if alpha is not None:
        flow_image_alpha = np.ones(u.shape[:2] + (1,)) * alpha
        flow_image = np.concatenate((flow_image, flow_image_alpha), axis=2)

    if to_float:
        # divide by 255.0  and convert to floating point.
        flow_image = flow_image / 255.0

    return flow_image


def flow_to_color(flow_uv, clip_flow=None, convert_to_bgr=False, zero_gray_value=1.0,
                  alpha=None, to_float=False):
    """
    Expects a two-dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
        zero_gray_value (bool, optional): grayscale color of zero-flow.
            Default is 1 (white)
        alpha (float, optional): if provided, add an alpha channel populated
            with this value
        to_float (bool, optional): indicate whether to divide by 255 and return
            floating point numbers (default is False)

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
        flow_mag0 = np.hypot(flow_uv[..., 0], flow_uv[..., 1])
        flow_mag = np.clip(flow_mag0, 0, clip_flow)
        scl = flow_mag / (flow_mag0 + 1e-5)
        flow_uv *= scl

    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]
    # Check if flow_uv is empty: avoid error from np.max if empty flow field is provided

    if flow_uv.size > 0:
        rad = np.sqrt(np.square(u) + np.square(v))
        rad_max = np.max(rad)
        epsilon = 1e-5
        u = u / (rad_max + epsilon)
        v = v / (rad_max + epsilon)
    return flow_uv_to_colors_raw(u, v, convert_to_bgr, zero_gray_value=zero_gray_value,
                                 alpha=alpha, to_float=to_float)


def flow_uv_to_color(flow_u, flow_v, clip_flow=None, convert_to_bgr=False,
                     zero_gray_value=1.0, alpha=None, to_float=False):
    """
    Same as flow_to_color, except that u and v can be passed separately.

    Args:
        flow_u (np.ndarray): Flow U image of shape [H,W]
        flow_v (np.ndarray): Flow V image of shape [H,W]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
        zero_gray_value (bool, optional): grayscale color of zero-flow.
            Default is 1 (white)
        alpha (float, optional): if provided, add an alpha channel populated with
            this value
        to_float (bool, optional): indicate whether to divide by 255 and return
            floating point numbers (default is False)

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_u.ndim == 2, 'input flow U must have 2 dimensions'
    assert flow_v.ndim == 2, 'input flow V must have 2 dimensions'
    assert flow_u.shape == flow_v.shape, "flow U and flow V must have the same shape"
    if clip_flow is not None:
        flow_mag0 = np.hypot(flow_u, flow_v)
        flow_mag = np.clip(flow_mag0, 0, clip_flow)
        scl = flow_mag / (flow_mag0 + 1e-5)
        flow_u *= scl
        flow_v *= scl

        # flow_u = np.clip(flow_u, 0, clip_flow)
        # flow_v = np.clip(flow_v, 0, clip_flow)

    if flow_u.size > 0:  # Avoid error from np.max if empty flow field is provided
        rad = np.sqrt(np.square(flow_u) + np.square(flow_v))
        rad_max = np.max(rad)
        epsilon = 1e-5
        flow_u = flow_u / (rad_max + epsilon)
        flow_v = flow_v / (rad_max + epsilon)

    return flow_uv_to_colors_raw(flow_u, flow_v, convert_to_bgr=convert_to_bgr,
                                 zero_gray_value=zero_gray_value, alpha=alpha,
                                 to_float=to_float)


def flow_key_map(size: int = 101):
    # create a `size` x `size` map of all the flow colors.
    x = np.linspace(-1, 1, size)
    dx, dy = np.meshgrid(x, x)
    flow_map = flow_uv_to_color(dx, dy)
    return flow_map
