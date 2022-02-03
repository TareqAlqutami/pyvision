#!/usr/bin/env python

'''
Lucas-Kanade tracker
====================

Lucas-Kanade sparse optical flow with tracking.
It uses goodFeaturesToTrack for track initialization and back-tracking for match verification
between frames.

Usage
-----
python sparse_track.py --video_src [<video_source>]


Keys
----
ESC - exit
'''

# Python 2/3 compatibility
from __future__ import print_function
from argparse import ArgumentParser

import numpy as np
import cv2 as cv


lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

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
    
class App:
    def __init__(self, video_src, track_len=10, detect_interval=5):
        """Optical flow class with back tracking and verification

        Args:
            video_src (str): opencv supported video source such as camera index or file path 
            track_len (int, optional): length of feature histroical track. Defaults to 10.
            detect_interval (int, optional): detection interval. Defaults to 5 frames.
        """
        self.track_len = track_len
        self.detect_interval = detect_interval
        self.tracks = []
        self.cam = cv.VideoCapture(video_src)
        self.frame_idx = 0

    def run(self):
        while True:
            _ret, frame = self.cam.read()
            if not _ret:
                print("Could not read a frame")
                try:
                    self.cam.release()
                except:
                    pass
                break

            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            vis = frame.copy()

            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, _st, _err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, _st, _err = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1
                new_tracks = []
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks.append(tr)
                    cv.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
                self.tracks = new_tracks
                cv.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
                draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))
            
            # if detection interval is passed
            if self.frame_idx % self.detect_interval == 0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv.circle(mask, (x, y), 5, 0, -1)
                p = cv.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])


            self.frame_idx += 1
            self.prev_gray = frame_gray
            cv.imshow('lk_track', vis)

            ch = cv.waitKey(1)
            if ch == 27:
                break



if __name__ == '__main__':
    print(__doc__)
    # extract video source from args and parse it to number if it is an index
    arg_prarser = ArgumentParser(description="Sparse optical flow tracker")
    arg_prarser.add_argument(
        "--video_src", help="video srource. defaults to video capture 0", default="0")

    # get arguments
    args = arg_prarser.parse_args()
    video_src = args.video_src


    # if video source is number then convert str to number (index of video source)
    if video_src.isnumeric():
        video_src = int(video_src)

    print(f"selected video source: {video_src}")

    App(video_src).run()

    print("Done")
    cv.destroyAllWindows()