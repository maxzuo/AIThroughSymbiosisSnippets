import io
import os
import argparse
import datetime

import numpy as np
import cv2

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

import scipy.ndimage as ndimage

import matplotlib.pyplot as plt

import json

from tqdm import tqdm

from dist import dist

def dense_optical_flow(videopath, clusters=4):
    cap = cv2.VideoCapture(videopath)

    ret, first_frame = cap.read()

    prev = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros(first_frame.shape, dtype=np.uint8)

    flows = []

    # hsv[...,1] = 255
    ret, frame = cap.read()
    i = 0

    with tqdm(total=cap.get(cv2.CAP_PROP_FRAME_COUNT)) as pbar:
        while ret:
            pbar.update()
            # i += 1

            # if i > 500:
            #     break

            nxt = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev,nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

            """
            uncomment below to view optical flow
            """
            # hsv[...,0] = ang * 180 / np.pi / 2
            # hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

            # display_hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            # cv2.imshow("optical flow", hsv)


            flows.append([np.sum(mag), np.average(ang, weights=mag)])


            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break

            prev = nxt
            ret, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()

    flows = np.asarray(flows)
    min_flow = np.min(flows, axis=0)
    max_flow = np.max(flows, axis=0)
    flows = (flows - min_flow) / (max_flow - min_flow)

    kM = KMeans(n_clusters=clusters, random_state=101, n_init=100).fit(flows)

    return flows, (kM.cluster_centers_, kM.labels_)

def write_video(videopath:str, outfile:str, flows, centers, labels, k=4):
    colors = [(255,255,0), (0,0,255), (255,0,0), (0,255,0)]

    cap = cv2.VideoCapture(videopath)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(3)
    height = cap.get(4)

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(outfile, fourcc, fps, (int(width), int(height)))

    _, frame = cap.read()

    # flows = dist(flows, centers)

    # pca = PCA(n_components=2)
    # flows = pca.fit_transform(flows)

    # centers = dist(centers, centers)
    # centers = pca.transform(centers)


    scaler = MinMaxScaler()

    scaler.fit(np.vstack([flows, centers]))
    flows = scaler.transform(flows)
    centers = scaler.transform(centers)

    # print(np.min(flows), np.max(flows))
    # assert np.min(flows) == 0.
    # assert np.max(flows) == 1.


    plt.ion()
    fig = plt.figure()

    for i,flow in enumerate(flows):
        ret,frame = cap.read()
        # fig.gca().scatter(np.hstack([centers.T[0], np.array([flow[0]])]), np.hstack([centers.T[1], np.array([flow[1]])]), 100, [*range(k),labels[i]])
        fig.gca().scatter(centers.T[0], centers.T[1], 50, range(k))
        fig.gca().scatter((*(-np.ones(k)),flow[0],),(*(-np.ones(k)),flow[1],),500,[*range(k),labels[i]])
        plt.axis('off')
        plt.xlim(right=1.05)
        plt.xlim(left=-0.05)
        plt.ylim(top=1.05)
        plt.ylim(bottom=-0.05)
        fig.tight_layout()
        fig.canvas.draw()

        plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape((*reversed(fig.canvas.get_width_height()),3))
        plot = cv2.resize(plot, None, fx=0.5, fy=0.5)
        plot = cv2.cvtColor(plot, cv2.COLOR_RGB2BGR)
        # print(plot.shape)
        frame[0:plot.shape[0], 0:plot.shape[1]] = plot
        out.write(frame)
        fig.canvas.flush_events()
        fig.clear()

    out.release()
    cap.release()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-v", "--video", help="path to video file", type=str, required=True)
    parser.add_argument("-o", "--outfile", help="path to output file", type=str, default=None)
    parser.add_argument("-k", "--kclusters", help="number of clusters to use", type=int, default=4)

    args = parser.parse_args()


    flows, (centers, cluster_assignments) = dense_optical_flow(args.video, args.kclusters)


    np.save("centers.npy", centers)
    np.save("labels.npy", cluster_assignments.astype(np.uint8))

    write_video(args.video, args.outfile, flows, centers, cluster_assignments, k=4)

    cv2.destroyAllWindows()