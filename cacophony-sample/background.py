import numpy as np
import cv2
from os import listdir
from os.path import isfile, join

possible_animals = ["possum", "stoat", "rat", "other"]

class Movement:

    def __init__(self, directory, learningRate = 0.3, changePercentage = 0.01):
        self.learningRate = learningRate
        self.changePercentage = changePercentage
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        self.files = sorted([join(directory, f) for f in listdir(directory) if isfile(join(directory, f))])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.fgbg = None
        self.files = None
        self.changePercentage = None
        self.learningRate = None

    def getMovementImages(self):
        images = []
        for file in self.files:
            frame = cv2.imread(file)
            fgmask = self.fgbg.apply(frame, self.learningRate)
            thresh = cv2.threshold(fgmask, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            _, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            change = frame.shape[0] * frame.shape[1] * self.changePercentage
            required_change = frame.shape[0] * frame.shape[1] * self.changePercentage
            cnt_required_change = frame.shape[0] * frame.shape[1] * 0.000000001
            # print("required area: " + str(required_change))
            total_change = 0
            for c in cnts:
                if len(c) != 0:
                    area = cv2.contourArea(c)
                    if area > cnt_required_change:
                        total_change += area
            foreground = cv2.bitwise_and(frame, frame, mask=fgmask) # This gets the foreground of the image. We could switch it over so the thing learns from these images
            # cv2.imshow("Foreground", foreground)
            # cv2.waitKey(2)
            # print("total change: " + str(total_change))
            if total_change > required_change:
                clazz = [x for x in possible_animals if x in file.lower()][0]
                images.append((clazz, file))
                # print("class: " + clazz)
            else:
                # print("class: nothing")
                images.append(("nothing", file))
        return images


if __name__ == "__main__":
    m = Movement("C:\\Users\\Matthew\\Downloads\\images_small\\p26_possum")
    for thing in m.getMovementImages():
        for thing2 in thing:
            for thing3 in thing2:
                print(thing3)