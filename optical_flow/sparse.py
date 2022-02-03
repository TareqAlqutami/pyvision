#!/usr/bin/env python

'''
Sparse optical flow algorithms

USAGE:  python sparse.py --algorithm [<algorithm>] --video_src [<video_source>]
        algorithm options: lucas-kanade
Keys:
 1   - toggle HSV flow visualization
 c   - Clean the image of the motion vectors
 ESC - exit

NOTES: this code doesn't handle if optical flow vectors are not found. If this happens, the code will exit

'''

# Python 2/3 compatibility
from __future__ import print_function
from argparse import ArgumentParser

import numpy as np
import cv2 as cv




# Shi-Tomasi corner detection parameters
feature_params = {
    'maxCorners': 500,
    'qualityLevel': 0.3,
    'minDistance': 7, 
    'blockSize': 7
    }

# Lucas-Kanade optical flow parameters
lk_params = {
    'winSize': (15, 15),
    'maxLevel': 2,
    'criteria': (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03),
}


def draw_str(dst, target:tuple, s:str):
    """Draw text over an image frame

    Args:
        dst : image frame
        target (tuple): target location(x,y) to draw
        s (str): text to write
    """
    x, y = target
    cv.putText(dst, s, (x+1, y+1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)
    
if __name__ == '__main__':
    print(__doc__)
    # extract video source from args and parse it to number if it is an index
    arg_prarser = ArgumentParser(description="Sparse optical flow")
    arg_prarser.add_argument(
        "--video_src", help="video srource. defaults to video capture 0", default="0")
    arg_prarser.add_argument(
        "--algorithm", help="Sparse optical flow algorithm to use", default="lucas-kanade")

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


    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    # flags to show hsv and glitch frames
    show_hsv = False

    # Take first frame and find corners in it
    _ret, prev_img = cam.read()
    if not _ret:
        cam.release()
        exit()
    prevgray = cv.cvtColor(prev_img, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(prevgray, mask=None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(prev_img)

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
            # calculate optical flow vectors based on  lucas-kanade algorithm
            p1, st, err = cv.calcOpticalFlowPyrLK(
                prevgray, gray, p0, None, **lk_params)
        else:
            raise ValueError(
                "Algorithm provided is not supported. Allowed options are: lucas-kanade")
        # no features found
        if p1 is None or len(p1)==0:
            print("no motion vector found for this frame. Exiting")
            exit()

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.astype(int).ravel()
            c, d = old.astype(int).ravel()
            mask = cv.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            mask = cv.circle(mask, (a, b), 5, color[i].tolist(), -1)
        vis = cv.add(img.copy(), mask)
        draw_str(vis, (20, 20), 'feature count: %d' % len(good_new))
        cv.imshow("frame", vis)

        # check keyboard keys, ESC: exit, `1`: on/off hsv visualization
        ch = cv.waitKey(10)
        if ch == 27:
            break

        # update history
        mask = np.zeros_like(prev_img)
        prev_img = img
        prevgray = gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    print('Cleaning..')
    try:
        cam.release()
    except:
        pass
    cv.destroyAllWindows()
