#!/usr/bin/env python3
""" Convolution with Channels"""


import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """A convolution on images with channels:
    images is a numpy.ndarray with shape (m, h, w, c) containing multiple
    images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
        c is the number of channels in the image
    kernel is a numpy.ndarray with shape (kh, kw, c) containing the kernel
    for the convolution
        kh is the height of the kernel
        kw is the width of the kernel
    padding is either a tuple of (ph, pw), ‘same’, or ‘valid’
        if ‘same’, performs a same convolution
        if ‘valid’, performs a valid convolution
        if a tuple:
            ph is the padding for the height of the image
            pw is the padding for the width of the image
    the image should be padded with 0’s
    stride is a tuple of (sh, sw)
        sh is the stride for the height of the image
        sw is the stride for the width of the image
    Returns: a numpy.ndarray containing the convolved images"""
    m, h, w, c = images.shape
    kh, kw = kernel.shape[:-1]
    sh, sw = stride

    img = images

    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
        convolveImage = np.zeros(shape=(m, h, w))
        img = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                     mode='constant',
                     constant_values=0)
    if padding == 'valid':
        h, w = int((h - kh) / sh + 1), int((w - kw) / sw + 1)
        convolveImage = np.zeros(shape=(m, h, w))
    if type(padding) is tuple:
        ph, pw = padding
        h, w = (h + 2 * ph - kh) // sh + 1, (w + 2 * pw - kw) // sw + 1
        convolveImage = np.zeros((m, h, w))
        img = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                     mode='constant',
                     constant_values=0)

    for y in range(h):
        for x in range(w):
            y0 = y * sh
            y1 = y0 + kh
            x0 = x * sw
            x1 = x0 + kw
            convolveImage[:, y, x] = np.sum(img[:, y0:y1, x0:x1, :] * kernel,
                                            axis=(1, 2, 3))
    return convolveImage
