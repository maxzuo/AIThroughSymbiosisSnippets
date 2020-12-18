import cv2
import time
import numpy as np

protoFile = "model/pose_deploy.prototxt"
weightsFile = "model/model.caffemodel"
nPoints = 22

frame = cv2.imread("hand_@.jpg")
frameCopy = np.copy(frame)

threshold = 0.1

frameHeight, frameWidth, *_ = frame.shape

aspect_ratio = frameWidth/frameHeight

inHeight = 368
inWidth = int(((aspect_ratio*inHeight)*8)//8)

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                  (0, 0, 0), swapRB=False, crop=False)

net.setInput(inpBlob)

output = net.forward()
output_shape = output.shape[-2:]
print(output.shape, _)

points = []

for i in range(nPoints):
    # confidence map of corresponding body's part.
    probMap = output[0, i, :, :]
    # probMap = cv2.resize(probMap, (frameWidth, frameHeight))

    # Find global maxima of the probMap.
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

    if prob > threshold :
        cv2.circle(frameCopy, (int(point[0] / output_shape[1] * frameWidth), int(point[1] / output_shape[0] * frameHeight)), 4, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        # cv2.putText(frameCopy, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

        # Add the point to the list if the probability is greater than the threshold
        points.append((int(point[0]), int(point[1])))
    else:
        points.append(None)
        #cv2.imshow('Output-Keypoints', frameCopy)

cv2.imwrite('frameCopy.jpg', frameCopy)