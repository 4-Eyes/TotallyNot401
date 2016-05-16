import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
from PIL import Image
import numpy

possible_animals = ["possum", "stoat", "rat", "other", "nothing"]

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

    def getMovementImages(self, scale):
        images = []
        frames = []
        percent = 1/len(self.files)
        for i,file in enumerate(self.files):
            print("Importing images...", str(round(percent*i*100)) + '%', end='\r')
            frame = cv2.imread(file)
            pt1 = (int(369 / frame.shape[1] * 640), int(445 / frame.shape[0] * 480))
            pt2 = (int(581 / frame.shape[1] * 640), int(474 / frame.shape[0] * 480))
            cv2.rectangle(frame, pt1, pt2, (0,0,0), -1)
            fgmask = self.fgbg.apply(frame, self.learningRate)

            frames.append(frame)

        print("Importing images... 100%")
        bg_image = self.fgbg.getBackgroundImage()
        bg_image = cv2.medianBlur(bg_image, 5)
        
        size = frames[0].shape[1], frames[0].shape[0]
        clazzes = []
        for i, frame in enumerate(frames):
            print("Analysing images...", str(round(percent*i*100)) + '%', end='\r')
            frame = cv2.medianBlur(frame, 5)
            diff = cv2.absdiff(bg_image, frame)
            thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)


            thresh = cv2.cvtColor(thresh, cv2.COLOR_RGB2GRAY)

            _, cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            change = frame.shape[0] * frame.shape[1] * self.changePercentage
            required_change = frame.shape[0] * frame.shape[1] * self.changePercentage
            total_change = 0
            for c in cnts:
                if len(c) != 0:
                    total_change += cv2.contourArea(c)

            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(frame).convert('L')
            #pil.show()
            pil.thumbnail((pil.size[0] * scale, pil.size[1] * scale), Image.ANTIALIAS)

            # extract the image as a vector of grayscale values between 0 and 1
            pil = (numpy.asarray(pil, dtype='float64') / 256.).reshape(size[0]*size[1])

            if total_change > required_change:
                clazz = [x for x in possible_animals if x in file.lower()][0]
                frames[i] = (clazz, pil, self.files[i])
                clazzes.append(clazz)
            else:
                frames[i] = ("nothing", pil, self.files[i])
                clazzes.append("nothing")
        print("Analysing images... 100%\nComplete")
        return frames, size, clazzes








            





        return images


        #for i in range(1,-1,-1):
        #    for file in self.files:
        #        frame = cv2.imread(file)

        #        pt1 = (int(369 / frame.shape[1] * 640), int(445 / frame.shape[0] * 480))
        #        pt2 = (int(581 / frame.shape[1] * 640), int(474 / frame.shape[0] * 480))

        #        cv2.rectangle(frame, pt1, pt2, (0,0,0), -1)

        #        cv2.imshow('foo', frame)

        #        fgmask = self.fgbg.apply(frame, self.learningRate * i)

        #        if i == 1:
        #            print('learning')
        #            continue
        #        elif i == 0:
        #            print('it was 0')

        #        thresh = cv2.threshold(fgmask, 25, 255, cv2.THRESH_BINARY)[1]
        #        thresh = cv2.dilate(thresh, None, iterations=2)
        #        _, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #        change = frame.shape[0] * frame.shape[1] * self.changePercentage
        #        required_change = frame.shape[0] * frame.shape[1] * self.changePercentage
        #        cnt_required_change = frame.shape[0] * frame.shape[1] * 0.000000001
        #        # print("required area: " + str(required_change))
        #        total_change = 0
        #        for c in cnts:
        #            if len(c) != 0:
        #                area = cv2.contourArea(c)
        #                if area > cnt_required_change:
        #                    total_change += area
        #        foreground = cv2.bitwise_and(frame, frame, mask=fgmask) # This gets the foreground of the image. We could switch it over so the thing learns from these images
        #        cv2.imshow("Foreground", foreground)
        #        cv2.waitKey(2)
        #        # print("total change: " + str(total_change))
        #        if total_change > required_change:
        #            clazz = [x for x in possible_animals if x in file.lower()][0]
        #            images.append((clazz, file))
        #            # print("class: " + clazz)
        #        else:
        #            # print("class: nothing")
        #            images.append(("nothing", file))
        #return images


if __name__ == "__main__":
    m = Movement("C:\\Users\\Matthew\\Downloads\\images_small\\p26_possum")
    for thing in m.getMovementImages():
        for thing2 in thing:
            for thing3 in thing2:
                print(thing3)