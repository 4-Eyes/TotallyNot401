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
        self.files = [join(directory, f) for f in listdir(directory) if isfile(join(directory, f))]

    def getMovementImages(self):
        images = []
        for file in self.files:
            frame = cv2.imread(file)
            fgmask = self.fgbg.apply(frame, self.learningRate)
            thresh = cv2.threshold(fgmask, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            _, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            change = frame.shape[0] * frame.shape[1] * self.changePercentage
            for c in cnts:
                if len(c) != 0 and cv2.contourArea(c) > change:
                    clazz = [x for x in possible_animals if x in file.lower()][0]
                    images.append((clazz, file))
                else:
                    images.append(("Nothing", file))
        return images

if __name__ == "__main__":
    m = Movement("C:\\Users\\Matthew\\Downloads\\images_small\\p26_possum")
    for thing in m.getMovementImages():
        for thing2 in thing:
            for thing3 in thing2:
                print(thing3)