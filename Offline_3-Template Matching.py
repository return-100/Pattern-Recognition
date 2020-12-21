import numpy as np
import cv2
from matplotlib import pyplot as plt

num_of_frames = 0
fps = 0

def process_video():
    global num_of_frames
    global fps
    video = cv2.VideoCapture("input.mov")
    fps = video.get(cv2.CAP_PROP_FPS)
    while True:
        ret, frame = video.read()
        if ret:
            cv2.imwrite("./frame/" + str(num_of_frames) + ".jpg", frame)
            num_of_frames += 1
        else:
            break
    video.release()
    cv2.destroyAllWindows()


def exhaustive_search(p):
    referenceImage = cv2.imread('reference.jpg', 1)
    M = referenceImage.shape[0]
    N = referenceImage.shape[1]
    currentI, currentJ = 0, 0
    outputFrame = []
    total_num_of_search = 0
    for frame in range(num_of_frames):
        templateImage = cv2.imread('./frame/' + str(frame) + '.jpg', 1)
        I = templateImage.shape[0]
        J = templateImage.shape[1]
        minDist = np.inf
        if frame == 0:
            for i in range(I - M + 1):
                for j in range(J - N + 1):
                    tempDist = templateImage[i: i + M, j: j + N].astype(np.int64)
                    ans = np.absolute(referenceImage.astype(np.int64) - tempDist)
                    tempDist = np.sum(np.square(ans))
                    total_num_of_search += 1
                    if tempDist < minDist:
                        minDist = tempDist
                        currentI = i
                        currentJ = j
            image = cv2.rectangle(templateImage, (currentJ, currentI), (currentJ + N, currentI + M), (255, 0, 0))
            outputFrame.append(image)
        else:
            centerI = currentI
            centerJ = currentJ
            for i in range(max(0, centerI - p), min(I, centerI + p)):
                for j in range(max(0, centerJ - p), min(J, centerJ + p)):
                    if i + M > I or j + N > J:
                        break
                    total_num_of_search += 1
                    tempDist = templateImage[i: i + M, j: j + N].astype(np.int64)
                    ans = np.absolute(referenceImage.astype(np.int64) - tempDist)
                    tempDist = np.sum(np.square(ans))
                    if tempDist < minDist:
                        minDist = tempDist
                        currentI = i
                        currentJ = j
            image = cv2.rectangle(templateImage, (currentJ, currentI), (currentJ + N, currentI + M), (255, 0, 0))
            outputFrame.append(image)
    videoCodec = cv2.VideoWriter_fourcc(*"mp4v")
    output = cv2.VideoWriter("./exhaustive.mp4", videoCodec, fps, (J, I))
    for frame in outputFrame:
        output.write(frame)
    output.release()
    return total_num_of_search


def logarithmic_search(p):
    referenceImage = cv2.imread('reference.jpg', 1)
    M = referenceImage.shape[0]
    N = referenceImage.shape[1]
    (currentI, currentJ) = (0, 0)
    (centerI, centerJ) = (0, 0)
    outputFrame = []
    total_num_of_search = 0
    for frame in range(num_of_frames):
        templateImage = cv2.imread('./frame/' + str(frame) + '.jpg', 1)
        I = templateImage.shape[0]
        J = templateImage.shape[1]
        minDist = np.inf
        if frame == 0:
            for i in range(I - M + 1):
                for j in range(J - N + 1):
                    tempDist = templateImage[i: i + M, j: j + N].astype(np.int64)
                    ans = np.absolute(referenceImage.astype(np.int64) - tempDist)
                    tempDist = np.sum(np.square(ans))
                    total_num_of_search += 1
                    if tempDist < minDist:
                        minDist = tempDist
                        currentI = i
                        currentJ = j
        else:
            currentP = p
            while True:
                minDist = np.inf
                k = np.ceil(np.log2(currentP))
                d = int(np.power(2, k - 1))
                if d < 1:
                    break
                for i in range(max(0, centerI - d), min(I, centerI + d + 1), d):
                    for j in range(max(0, centerJ - d), min(J, centerJ + d + 1), d):
                        if i + M > I or j + N > J:
                            break
                        total_num_of_search += 1
                        tempDist = templateImage[i: i + M, j: j + N].astype(np.int64)
                        ans = np.absolute(referenceImage.astype(np.int64) - tempDist)
                        tempDist = np.sum(np.square(ans))
                        if tempDist < minDist:
                            minDist = tempDist
                            currentI = i
                            currentJ = j
                centerI = currentI
                centerJ = currentJ
                currentP = int(currentP / 2)
        image = cv2.rectangle(templateImage, (currentJ, currentI), (currentJ + N, currentI + M), (255, 0, 0))
        outputFrame.append(image)
        centerI = currentI
        centerJ = currentJ
    videoCodec = cv2.VideoWriter_fourcc(*"mp4v")
    output = cv2.VideoWriter("./logarithmic.mp4", videoCodec, fps, (J, I))
    for frame in outputFrame:
        output.write(frame)
    output.release()
    return total_num_of_search


def hierarchical_search(p):
    referenceImage = cv2.imread('reference.jpg', 1)
    M = referenceImage.shape[0]
    N = referenceImage.shape[1]
    (currentI, currentJ) = (0, 0)
    (centerI, centerJ) = (0, 0)
    outputFrame = []
    total_num_of_search = 0
    for frame in range(num_of_frames):
        templateImage = cv2.imread('./frame/' + str(frame) + '.jpg', 1)
        if frame == 0:
            I = templateImage.shape[0]
            J = templateImage.shape[1]
            minDist = np.inf
            for i in range(I - M + 1):
                for j in range(J - N + 1):
                    tempDist = templateImage[i: i + M, j: j + N].astype(np.int64)
                    ans = np.absolute(referenceImage.astype(np.int64) - tempDist)
                    tempDist = np.sum(np.square(ans))
                    total_num_of_search += 1
                    if tempDist < minDist:
                        minDist = tempDist
                        currentI = i
                        currentJ = j
        else:
            currentP = [int(p / 4), 1, 1]
            centerI = int(centerI / 4)
            centerJ = int(centerJ / 4)
            lowPassReference = [referenceImage]
            lowPassTemplate = [templateImage]
            for i in range(2):
                lowPassReference.append(cv2.pyrDown(lowPassReference[i]))
                lowPassTemplate.append((cv2.pyrDown(lowPassTemplate[i])))
            for level in range(3):
                minDist = np.inf
                (curI, curJ) = (lowPassTemplate[2 - level].shape[0], lowPassTemplate[2 - level].shape[1])
                (curM, curN) = (lowPassReference[2 - level].shape[0], lowPassReference[2 - level].shape[1])
                for i in range(max(0, centerI - currentP[level]), min(curI, centerI + currentP[level] + 1), currentP[level]):
                    for j in range(max(0, centerJ - currentP[level]), min(curJ, centerJ + currentP[level] + 1), currentP[level]):
                        if i + curM > curI or j + curN > curJ:
                            break
                        total_num_of_search += 1
                        tempDist = lowPassTemplate[2 - level][i: i + curM, j: j + curN].astype(np.int64)
                        ans = np.absolute(lowPassReference[2 - level].astype(np.int64) - tempDist)
                        tempDist = np.sum(np.square(ans))
                        if tempDist < minDist:
                            minDist = tempDist
                            currentI = i
                            currentJ = j
                centerI = currentI * 2
                centerJ = currentJ * 2
        image = cv2.rectangle(templateImage, (currentJ, currentI), (currentJ + N, currentI + M), (255, 0, 0))
        outputFrame.append(image)
        centerI = currentI
        centerJ = currentJ
    videoCodec = cv2.VideoWriter_fourcc(*"mp4v")
    output = cv2.VideoWriter("./hierarchical.mp4", videoCodec, fps, (J, I))
    for frame in outputFrame:
        output.write(frame)
    output.release()
    return total_num_of_search


if __name__ == "__main__":
    report = open("report.txt", "w")
    report.write("P     Exhaustive          2D Log          Hierarchical\n")
    process_video()
    for p in [4, 5, 7, 8, 12, 15]:
        report.write(str(p) + "     ")
        num_of_search = exhaustive_search(p)
        report.write(str(int(num_of_search / num_of_frames)) + "          ")
        num_of_search = logarithmic_search(p)
        report.write(str(int(num_of_search / num_of_frames)) + "          ")
        num_of_search = hierarchical_search(p)
        report.write(str(int(num_of_search / num_of_frames)) + "\n")
