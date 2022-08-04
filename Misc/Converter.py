# Import the required libraries.
import os
import numpy as np
from glob import glob
from nibabel import load, save, Nifti1Image


# Create the class.
class Convert(object):
    # Class Constructor
    def __init__(self, directory=" "):
        self.dir = directory
        self.fileList = [os.path.basename(x) for x in glob(self.dir + '*.mnc')]
        self.affine = np.array([
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1]])

    # Create the converter function.
    def conversion(self):
        for i in range(len(self.fileList)):
            minc = load(self.dir + self.fileList[i])
            basename = self.fileList[i].split(os.extsep, 1)[0]
            out = Nifti1Image(minc.get_fdata(), affine=self.affine)
            save(out, self.dir + basename + '.nii.gz')
            print("Converting: ", basename)
