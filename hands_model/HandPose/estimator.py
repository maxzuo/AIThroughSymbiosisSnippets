import cv2
import numpy as np
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from tqdm import tqdm

class HandEstimator():

    def __init__(self, protoFile="model/pose_deploy.prototxt", weightsFile="model/model.caffemodel", threshold=0.1):
        self.protofile = protoFile
        self.weightsFile = weightsFile
        self.threshold = threshold

        self.nPoints = 22

        self.net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


    # TODO: no hand assignment yet
    def predict(self, image, n=5):
        height, width, *_ = image.shape
        aspect_ratio = width/height

        inHeight = 368
        inWidth = int((aspect_ratio * inHeight * 8)//8)

        blob = cv2.dnn.blobFromImage(image, 1.0 / 255, (inWidth, inHeight),
                                  (0, 0, 0), swapRB=False, crop=False)

        self.net.setInput(blob)
        output = self.net.forward()[0]
        outputHeight, outputWidth = output.shape[-2:]

        res = {}

        for i,prob_map in enumerate(output[:-1]):
            res[i] = []
            point_max = filters.maximum_filter(prob_map, n)
            maxima = (prob_map == point_max)
            point_min = filters.minimum_filter(prob_map, n)
            diff = ((point_max - point_min) > self.threshold)
            maxima[diff == 0] = 0

            labeled, _ = ndimage.label(maxima)
            slices = ndimage.find_objects(labeled)

            for dy,dx in slices:
                x_center = (dx.start + dx.stop - 1) / 2
                y_center = (dy.start +  dy.stop - 1) / 2
                res[i].append((x_center / outputWidth * width, y_center / outputHeight * height))
        return res


# testing purposes only
if __name__ == "__main__":
    estimator = HandEstimator()
    # image = cv2.imread('../hand_1.jpg')
    # points = estimator.predict(cv2.imread('../hand_1.jpg'))
    # import json
    # print(json.dumps(points, indent=4))

    cap = cv2.VideoCapture('../../video/test.mp4')

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(3))
    height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    length = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    out = cv2.VideoWriter('../red3.mp4', fourcc, fps, (int(width), int(height)))
    for i in tqdm(range(int(length // 2)+10)):
        ret, frame = cap.read()
        if not ret:
            break
        f = frame[width//3:width//3 * 2, height // 5 * 3: height]
        
        points = estimator.predict(f)
        for key, points in points.items():
            for x,y in points:
                cv2.circle(f, (int(x), int(y)), 4, (0, 255, 255), -1)
        frame[width//3:width//3 * 2, height // 5 * 3: height] = f
        out.write(frame)
    out.release()
    cap.release()
    # cv2.imwrite('../output.jpg', image)