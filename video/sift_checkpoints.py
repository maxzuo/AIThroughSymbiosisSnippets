import os
import argparse
import datetime

import numpy as np
import cv2
import tqdm

from sklearn.cluster import MiniBatchKMeans
import scipy.ndimage as ndimage

sec_step = 0.2

def detect_sift(video_path, k=600):
    sift = cv2.SIFT_create()

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    length = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    sec = 0.0
    d_by_frame = []
    index = 0
    selected_index = 0
    print("Detecting features")
    for _ in tqdm.tqdm(range(int(length / fps / sec_step))):

        cap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        sec += sec_step
        ret, frame = cap.read()

        if not ret:
            break
        key = cv2.waitKey(1) & 0xFF
        cv2.imshow("frame",frame)
        if key == ord('q'):
            break
        elif key == ord('b'):
            selected_index = index
        # cap.
        frame = cv2.resize(frame, (256, 256))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray,None)
        d_by_frame.append(des)
        # gray = cv2.drawKeypoints(gray, kp, gray)

        index += 1
    cap.release()
    cv2.destroyAllWindows()
    # produce vocabulary
    print("Building vocabulary")
    km = MiniBatchKMeans(n_clusters=k, random_state=101, max_iter=1_000).fit(np.vstack(d_by_frame))
    vocabulary = km.cluster_centers_

    frame_histograms = [np.bincount(km.predict(descriptor), minlength=k) for descriptor in d_by_frame]

    return vocabulary, frame_histograms, frame_histograms[selected_index]

def similarities(hist, hists):
    sim = np.asarray([(hist @ h) / (1e-6 + np.linalg.norm(hist) * np.linalg.norm(h)) for h in hists])
    return sim

def write_queried_video(videofile, outfile, mask):
    cap = cv2.VideoCapture(videofile)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(3)
    height = cap.get(4)

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(outfile, fourcc, fps, (int(width), int(height)))

    frames_indices = np.argwhere(mask == 1)
    ret = True

    for index in tqdm.tqdm(frames_indices):
        # print(int(index * sec_step * 1000))
        cap.set(cv2.CAP_PROP_POS_MSEC, int(index * sec_step * 1000))
        for i in range(int(fps*sec_step)):
            ret, frame = cap.read()

            sec = index*sec_step + i/fps
            cv2.putText(frame, "%02d:%02d:%02d.%03d" % (sec // 360 % 60, sec // 60 % 60, sec % 60, sec % 1), (0,0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

            out.write(frame)

    out.release()
    cap.release()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-v", "--video", help="path to video file", type=str, required=True)
    parser.add_argument("-o", "--outfile", help="path to output file", type=str, default=None)
    parser.add_argument("-k", "--kclusters", help="number of clusters to use", type=int, default=600)

    args = parser.parse_args()

    vocab, hists, first_hist = detect_sift(args.video, k=args.kclusters)

    # first_hist = hists[15]

    sim = similarities(first_hist, hists)

    print("querying")
    frames = ndimage.maximum_filter1d((sim > 0.5).astype(np.uint8), 4)

    print("writing queried frames")
    write_queried_video(args.video, args.outfile, frames)