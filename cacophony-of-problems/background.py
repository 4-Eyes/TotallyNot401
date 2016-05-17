import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
from PIL import Image
import numpy

possible_animals = ["possum", "stoat", "rat", "other", "nothing"]

class Movement:

    def __init__(self, directory, learningRate = 0.3, changePercentage = 0.0005, displayOpenCVImage = True):
        self.learningRate = learningRate
        self.changePercentage = changePercentage
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        self.files = sorted([join(directory, f) for f in listdir(directory) if isfile(join(directory, f))])
        self.display_opencv_image = displayOpenCVImage
        print("-> Working on subdirectory",directory)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.fgbg = None
        self.files = None
        self.changePercentage = None
        self.learningRate = None

    def getMovementImages(self, scale):
        images = []
        frames = []
        percent = 1/len(self.files)
        for i,file in enumerate(self.files):
            print("Importing images...", str(round(percent*i*100)) + '%', end='\r')
            frame = cv2.imread(file)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = self.clahe.apply(frame)
            frame = cv2.medianBlur(frame, 5)
            pt1 = (int(369 / frame.shape[1] * 640), int(445 / frame.shape[0] * 480))
            pt2 = (int(581 / frame.shape[1] * 640), int(474 / frame.shape[0] * 480))
            cv2.rectangle(frame, pt1, pt2, (0,0,0), -1)
            fgmask = self.fgbg.apply(frame, self.learningRate)
            frames.append(frame)

        print("Importing images... 100%")
        bg_image = self.fgbg.getBackgroundImage()
        
        h = frames[0].shape[0]
        size = int(frames[0].shape[1]*scale), int(frames[0].shape[0]*scale)
        clazzes = []
        clazzes_unique = set()
        for i, frame in enumerate(frames):
            file = self.files[i]
            print("Analysing images...", str(round(percent*i*100)) + '%', end='\r')
            diff = cv2.absdiff(bg_image, frame)
            thresh = cv2.GaussianBlur(diff, (21, 21), 0)
            thresh = cv2.threshold(thresh, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)

            _, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            change = frame.shape[0] * frame.shape[1] * self.changePercentage
            required_change = frame.shape[0] * frame.shape[1] * self.changePercentage
            total_change = 0
            for c in [x for x in cnts if len(x) != 0]:
                total_change += cv2.contourArea(c)

            pil = Image.fromarray(diff).convert('L')
            pil.thumbnail((pil.size[0] * scale, pil.size[1] * scale), Image.ANTIALIAS)

            # extract the image as a vector of grayscale values between 0 and 1
            pil = (numpy.asarray(pil, dtype='float64') / 256.).reshape(size[0]*size[1])

            if total_change > required_change:
                clazz = [x for x in possible_animals if x in file.lower()][0]
            else:
                clazz = "nothing"

            if self.display_opencv_image:
                disp = diff.copy()
                cv2.putText(disp, clazz, (0,h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)
                cv2.imshow('Classed As',disp)
                cv2.imshow('Edges',thresh)
                cv2.waitKey(1)

            frames[i] = pil
            clazzes.append(clazz)
            if not clazz in clazzes_unique:
                clazzes_unique.add(clazz)

        print("Analysing images... 100%\nSubdirectory Complete\n")
        return (frames, clazzes), size, clazzes_unique

if __name__ == "__main__":
    m = Movement("C:\\Users\\Matthew\\Downloads\\images_small\\p26_possum")
    for thing in m.getMovementImages():
        for thing2 in thing:
            for thing3 in thing2:
                print(thing3)