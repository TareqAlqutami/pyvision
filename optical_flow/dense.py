#!/usr/bin/env python

'''
Dense optical flow algorithms

USAGE:  python dense.py --algorithm [<algorithm>] --video_src [<video_source>]
        algorithm options: lucas-kanade, farneback (default), and rlof
Keys:
 1   - toggle HSV flow visualization
 ESC - exit

NOTES: this code doesn't handle if optical flow vectors are not found. If this happens, the code will exit

'''

# Python 2/3 compatibility
from __future__ import print_function
from argparse import ArgumentParser

import numpy as np
import cv2 as cv


# LK dense algorithms parameters
lk_dense_params = {
    # stride used in sparse match computation. Lower values usually result in higher quality but slow down the algorithm.
    'grid_step': 8,
    # number of nearest-neighbor matches considered, when fitting a locally affine model. Lower values makes algorithm faster but degrades quality.
    'k': 128,
    # how fast the weights decrease in the locally-weighted affine fitting. Higher values can help preserve fine details.
    'sigma': 0.05,
    # defines whether the fastGlobalSmootherFilter is used for post-processing after interpolation
    'use_post_proc': True,
}

# farneback algorithm parameters
farneback_params = {
    # pyr_scale=0.5 means a classical pyramid, where each next layer is twice smaller than the previous one.
    'pyr_scale': 0.5,
    # number of pyramid layers including the initial image; levels=1 means that no extra layers are created and only the original images are used.
    'levels': 3,
    # averaging window size; larger values increase the algorithm robustness to image noise and give more chances for fast motion detection, but yield more blurred motion field.
    'winsize': 15,
    'iterations': 3,  # number of iterations the algorithm does at each pyramid level
    'poly_n': 5,  # size of the pixel neighborhood used to find polynomial expansion in each pixel; bigger means more robust but blurred motion field. typically =5 or =7
    'poly_sigma': 1.2,  # standard deviation of the Gaussian that is used to smooth derivatives used as a basis for the polynomial expansion
    'flags': 0
}


def draw_flow(in_img, flow, step=16):
    """
    Draw flow vectors on the BGR image

    Args:
        in_img (np.array): cv colored image
        flow (np.array): 2d dense motion vectors to plot
        step (int, optional): step size. Defaults to 16.

    Returns:
        img: cv image with flow vectors
    """
    img = in_img.copy()
    h, w = img.shape[:2]
    # create a grid with step size `step` and limited by image wxh
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)

    # create line vectors from each grid point
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    # plot lines into the image
    cv.polylines(img, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv.circle(img, (x1, y1), 1, (0, 255, 0), -1)
    return img


def draw_hsv(flow):
    """
    Visualize motion vectors as HSV image where direction is hue and  value is magnitude 
    Args:
        flow (np.array): 2d dense optical flow vectors

    Returns:
        np.array: BGR cv image
    """
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]

    # convert to angle and magnitude
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)

    # create hsv image
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang*(180/np.pi/2)
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(v*4, 255)

    # convert to BGR
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return bgr


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv.remap(img, flow, None, cv.INTER_LINEAR)
    return res


if __name__ == '__main__':
    print(__doc__)
    # extract video source from args and parse it to number if it is an index
    arg_prarser = ArgumentParser(description="Dense optical flow")
    arg_prarser.add_argument(
        "--video_src", help="video srource. defaults to video capture 0", default="0")
    arg_prarser.add_argument(
        "--algorithm", help="Dense optical flow algorithm to use", default="farneback")

    # get arguments
    args = arg_prarser.parse_args()
    video_src = args.video_src
    algorithm = args.algorithm

    # if video source is number then convert str to number (index of video source)
    if video_src.isnumeric():
        video_src = int(video_src)

    print(f"selected video source: {video_src}")
    print(f"selected algorithm: {algorithm}")

    # create video source and read first frame
    cam = cv.VideoCapture(video_src)
    _ret, prev_img = cam.read()
    if not _ret:
        cam.release()
        exit()
    prevgray = cv.cvtColor(prev_img, cv.COLOR_BGR2GRAY)

    # flags to show hsv and glitch frames
    show_hsv = False

    # loop forever
    while True:
        # read and convert frame to gray scale
        _ret, img = cam.read()

        if not _ret:
            print("Did not get any frame!")
            cam.release()
            break
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        if algorithm == "lucas-kanade":
            # calculate optical flow vectors based on dense lucas-kanade algorithm
            flow = cv.optflow.calcOpticalFlowSparseToDense(
                prevgray, gray, None, **lk_dense_params)

        elif args.algorithm == 'farneback':
            # The main idea of this method is to approximate some neighbors of each pixel with a polynomial
            flow = cv.calcOpticalFlowFarneback(
                prevgray, gray, None, **farneback_params)

        elif args.algorithm == "rlof":
            # The main idea of this work is that the intensity constancy assumption doesn’t fully reflect how the real world behaves. 
            # There are also shadows, reflections, weather conditions, ..
            # The RLOF algorithm is based on an illumination model proposed by Gennert and Negahdaripour in 1995
            flow = cv.optflow.calcOpticalFlowDenseRLOF(prev_img, img, None)

        else:
            raise ValueError(
                "Algorithm provided is not supported. Allowed options are: lucas-kanade, farneback, and rlof")

        # update history
        prevgray = gray
        prev_img = img

        # draw image with a grid of motion vectors
        cv.imshow('flow', draw_flow(img, flow))

        # show motion vectors as hsv hues and values if enabled
        if show_hsv:
            cv.imshow('flow HSV', draw_hsv(flow))

        # check keyboard keys, ESC: exit, `1`: on/off hsv visualization
        ch = cv.waitKey(5)
        if ch == 27:
            break
        if ch == ord('1'):
            show_hsv = not show_hsv
            print('HSV flow visualization is', ['off', 'on'][show_hsv])

    print('Cleaning..')
    try:
        cam.release()
    except:
        pass
    cv.destroyAllWindows()
