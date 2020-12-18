import os
import argparse
import datetime

import numpy as np
import cv2
import tqdm

from sklearn.cluster import KMeans
import scipy.ndimage as ndimage

import matplotlib.pyplot as plt

from dist import new_point

sec_step = 0.2

def lk_optical_flow(videopath):
    cap = cv2.VideoCapture(videopath)

    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)
    lk_params = dict(winSize=(15,15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    ret, first_frame = cap.read()
    frame = first_frame
    old_gray = first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(first_frame, **feature_params)

    flow = []

    plt.ion()
    fig = plt.figure()
    # mag_fig = plt.figure()
    max_mag = 10


    while ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **lk_params)

        good_new = p1[st==1]
        good_old = p0[st==1]


        old_gray = gray
        p0 = good_new.reshape(-1,1,2)
        updated_corners = cv2.goodFeaturesToTrack(gray, **feature_params)
        new_points = new_point(p0.reshape((-1,2)), updated_corners.reshape((-1,2)), atol=1)

        # assert new_points.shape, updated_corners.shape
        new_corners = updated_corners[new_points]
        p0 = np.vstack((p0, new_corners))

        for i in range(len(good_new)):
            new, old = good_new[i], good_old[i]
            cv2.line(frame, (*new,), (*old,), (0, 255, 0), 1)

        for corner in new_corners:
            corner = (*corner.ravel(),)
            cv2.circle(frame, (*corner,), 3, (255, 0, 0), 2)

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

        # calculate directions
        diff = good_new - good_old
        # print(diff.shape)
        directions = np.arctan2(diff.T[0], diff.T[1])
        magnitudes = np.linalg.norm(diff, axis=1)
        # assert directions.shape, magnitudes.shape

        # direct_hist, bin_edges = np.histogram(directions, bins=8, range=(-np.pi, np.pi))
        # update histogram map

        # draw directions
        fig.gca().hist(directions, bins=20, range=(-np.pi, np.pi), weights=magnitudes)
        fig.canvas.draw()
        fig.canvas.flush_events()
        fig.clear()

        # draw magnitudes
        mag = np.median(magnitudes)
        print(mag)

        ret, frame = cap.read()
    cap.release()








if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-v", "--video", help="path to video file", type=str, required=True)
    parser.add_argument("-o", "--outfile", help="path to output file", type=str, default=None)
    parser.add_argument("-k", "--kclusters", help="number of clusters to use", type=int, default=4)

    args = parser.parse_args()

    lk_optical_flow(args.video)

    cv2.destroyAllWindows()