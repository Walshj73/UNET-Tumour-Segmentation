# Import the required libraries.
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


# Class creation.
class Water_Shed(object):
    # Class constructor.
    def __init__(self, images):
        self.images = images

    # Method to convert the images to gray scale.
    def convertGrayscale(self):
        self.gray_array = []
        for i in range(len(self.images)):
            self.gray_images = cv2.cvtColor(self.images[i], cv2.COLOR_BGR2GRAY)
            self.gray_array.append(self.gray_images)

    # Method applies a highpass filter to the image.
    def highPassFilter(self):
        lowpass = ndimage.gaussian_filter(self.gray_image, 5)
        self.highpass_3x3 = self.gray_image - lowpass

        plt.subplot(121)
        plt.gca().set_title("Original")
        plt.imshow(self.image, cmap='gray')
        plt.xticks([]), plt.yticks([])

        plt.subplot(122)
        plt.gca().set_title("High Pass Filtered")
        plt.imshow(self.highpass_3x3, cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.show()

    # Method applies a median filter to the image.
    def medianFilter(self, sigma):
        self.median_array = []
        for i in range(len(self.gray_array)):
            self.median = cv2.medianBlur(self.gray_array[i], sigma)
            self.median_array.append(self.median)

        plt.subplot(121)
        plt.gca().set_title("Original")
        plt.imshow(self.images[0], cmap='gray')
        plt.xticks([]), plt.yticks([])

        plt.subplot(122)
        plt.gca().set_title("Median Blur")
        plt.imshow(self.median_array[0], cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.show()

    # Method applies a threshold algorithm to the median image.
    def thresholdSegment(self):
        self.thresh_array = []
        for i in range(len(self.median_array)):
            self.retval, self.thresh = cv2.threshold(
                    self.median_array[i],
                    180,
                    255,
                    cv2.THRESH_BINARY
            )
            self.thresh_array.append(self.thresh)

        plt.subplot(121)
        plt.gca().set_title("Original")
        plt.imshow(self.median_array[0], cmap='gray')
        plt.xticks([]), plt.yticks([])

        plt.subplot(122)
        plt.gca().set_title("Thresholded")
        plt.imshow(self.thresh_array[0], cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.show()

    # Method computes the watershed segmentation of the threshold.
    def waterShed(self):
        self.kernel = np.ones((3, 3), np.uint8)
        self.markers_array = []
        for i in range(len(self.thresh_array)):
            self.opening = cv2.morphologyEx(
                    self.thresh_array[i],
                    cv2.MORPH_OPEN,
                    self.kernel,
                    iterations=2
            )
            self.sure_bg = cv2.dilate(self.opening, self.kernel, iterations=3)
            self.dist_transform = cv2.distanceTransform(
                    self.opening,
                    cv2.DIST_L2,
                    5
            )
            self.ret, self.sure_fg = cv2.threshold(
                    self.dist_transform,
                    0.7 * self.dist_transform.max(),
                    255,
                    0
            )
            self.sure_fg = np.uint8(self.sure_fg)
            self.unknown = cv2.subtract(self.sure_bg, self.sure_fg)
            self.ret, self.markers = cv2.connectedComponents(self.sure_fg)
            self.markers = self.markers + 1
            self.markers[self.unknown == 255] = 0
            self.markers = cv2.watershed(self.images[i], self.markers)
            self.markers_array.append(self.markers)

        plt.subplot(121)
        plt.gca().set_title("Original")
        plt.imshow(self.images[0], cmap='gray')
        plt.xticks([]), plt.yticks([])

        plt.subplot(122)
        plt.gca().set_title("Watershed")
        plt.imshow(self.markers_array[0], cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.show()

    # Method computes the morphological operation of the image.
    def morphOperation(self):
        self.morph_array = []
        for i in range(len(self.markers_array)):
            self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            self.erode = cv2.erode(
                    self.thresh_array[i],
                    self.kernel,
                    iterations=2
            )
            self.morph_array.append(self.erode)

        plt.subplot(121)
        plt.gca().set_title("Original")
        plt.imshow(self.images[0], cmap='gray')
        plt.xticks([]), plt.yticks([])

        plt.subplot(122)
        plt.gca().set_title("Morphed")
        plt.imshow(self.morph_array[0], cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.show()

    # Method writes the final segment to a new image.
    def writeSegment(self, OUTDIR):
        for i in range(len(self.morph_array)):
            cv2.imwrite(
                    OUTDIR +
                    "Result_" +
                    str(i) +
                    ".png",
                    self.morph_array[i]
            )


# Main method.
if __name__ == "__main__":
    PATH = '../Evaluation Scans/Full/'
    OUTDIR = './Water Shed Results/Full/'

    images = [cv2.imread(file) for file in glob.glob(PATH + "*.png")]

    water = Water_Shed(images)
    water.convertGrayscale()
    # water.highPassFilter()
    water.medianFilter(5)
    water.thresholdSegment()
    water.waterShed()
    water.morphOperation()
    water.writeSegment(OUTDIR)
