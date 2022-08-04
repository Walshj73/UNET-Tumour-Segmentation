# Import the required libraries.
import os
import skimage.io
import nibabel as nib
from sklearn.model_selection import train_test_split


# Create the Load Data class.
class LoadData3D(object):
    # Class Constructor.
    def __init__(self, directoryImages=" ", directoryMask=" "):
        self.dirImages = directoryImages
        self.dirMask = directoryMask

    # Load a single image.
    def loadImage(self, dataFile):
        image = nib.load(self.dirImages + dataFile)
        return image

    # Load the images and store in a list.
    def loadImages(self):
        self.files = os.listdir(self.dirImages)
        data_all = []
        for i in self.files:
            self.images = nib.load(self.dirImages + i)
            data_all.append(self.images)
        return data_all

    # Load a single tumour mask.
    def loadMask(self, dataFile):
        mask = nib.load(self.dirMask + dataFile)
        return mask

    # Load the tumour masks and store them as a list.
    def loadMasks(self):
        self.files = os.listdir(self.dir)
        data_all = []
        for i in self.files:
            self.images = nib.load(self.dirMask + i)
            data_all.append(self.images)
        return data_all

    # Get the data of nibabel object.
    def getData(self, image):
        data = image.get_fdata()
        return data

    # Get the data of multiple nibabel objects.
    def retrieveData(self, images):
        imageData = []
        for i in range(len(images)):
            self.data = images[i].get_fdata()
            imageData.append(self.data)
        return imageData


# Class for 2-Dimensional data
class LoadData2D(object):
    # Read a single image.
    def readImage(self, PATH):
        return skimage.io.imread(PATH)

    # Read in multiple images.
    def readImages(self, PATH, extension):
        args = [os.path.join(PATH, filename)
                for filename in os.listdir(PATH)
                if any(filename.lower().endswith(ext) for ext in extension)]
        imgs = [self.readImage(arg) for arg in args]
        return imgs

    # X = Images and y = Masks (Must be arrays not objects!)
    def splitData(self, images, masks, splitSize):
        X_train, X_test, y_train, y_test = train_test_split(
                images,
                masks,
                test_size=splitSize
        )
        return X_train, X_test, y_train, y_test
