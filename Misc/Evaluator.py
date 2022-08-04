# Import the required libaries
import os
import glob
import numpy as np
import skimage.io
from skimage import color


# Evaluator class.
class Evaluate(object):
    # Class constructor, assigns prediction and ground truth.
    def __init__(self, pred_image, gt_image):
        self.pred_image = pred_image
        self.gt_image = gt_image

    def cm_terms(self):
        n11 = n12 = n21 = n22 = 0
        [rows, cols] = self.gt_image.shape

        for i in range(rows):
            for j in range(cols):
                if self.gt_image[i, j] == 0 and self.pred_image[i, j] == 0:
                    n11 = n11+1
                if self.gt_image[i, j] == 0 and self.pred_image[i, j] >= 1:
                    n12 = n12 + 1
                if self.gt_image[i, j] >= 1 and self.pred_image[i, j] == 0:
                    n21 = n21 + 1
                if self.gt_image[i, j] >= 1 and self.pred_image[i, j] >= 1:
                    n22 = n22 + 1

        return n11, n12, n21, n22

    # Returns the pixel accuracy.
    def pixel_accuracy(self):

        n11, n12, n21, n22 = Evaluate.cm_terms(self)
        t1 = n11 + n12
        t2 = n21 + n22

        if (t1+t2) == 0:
            pa = 0
        else:
            pa = float(n11+n22)/float(t1+t2)
        return pa

    # Returns the mean accuracy.
    def mean_accuracy(self):

        n11, n12, n21, n22 = Evaluate.cm_terms(self)
        t1 = n11 + n12
        t2 = n21 + n22

        if t1 == 0 and t2 != 0:
            ma = float(n22)/float(t2)
        if t1 != 0 and t2 == 0:
            ma = float(n11)/float(t1)
        if t1 == 0 and t2 == 0:
            ma = 0
        else:
            ma = (float(n11)/float(t1) + float(n22)/float(t2))/2
        return ma

    # Returns the IOU.
    def IOU(self):
        intersection = np.logical_and(self.gt_image, self.pred_image)
        union = np.logical_or(self.gt_image, self.pred_image)
        iou_score = np.sum(intersection) / np.sum(union)
        return iou_score

    # Returns the mean IOU.
    def mean_IOU(self):

        n11, n12, n21, n22 = Evaluate.cm_terms(self)
        t1 = n11 + n12
        t2 = n21 + n22

        if (t1+n21) == 0 and (t2+n12) != 0:
            mIOU = float(n22)/float(t2+n12)
        if (t1 + n21) != 0 and (t2 + n12) == 0:
            mIOU = float(n11)/float(t1+n21)
        else:
            mIOU = (float(n11)/float(t1+n21) + float(n22)/float(t2+n12))/2
        return mIOU

    # Returns the frequency weighted IOU.
    def fweight_IOU(self):

        n11, n12, n21, n22 = Evaluate.cm_terms(self)
        t1 = n11 + n12
        t2 = n21 + n22

        fwIOU = float(
                float(t1*n11)/float(n11+n12+n21) +
                float(t2*n22)/float(n12+n21+n22))/float(n11+n12+n21+n22)

        return fwIOU

    def roc(self):

        self.pred_image
        self.gt_image

        TP = TN = FP = FN = 0
        [rows, cols] = self.gt_image.shape

        for i in range(rows):
            for j in range(cols):
                if self.gt_image[i, j] >= 1 and self.pred_image[i, j] >= 1:
                    TP = TP + 1
                if self.gt_image[i, j] == 0 and self.pred_image[i, j] == 0:
                    TN = TN + 1
                if self.gt_image[i, j] == 0 and self.pred_image[i, j] >= 1:
                    FP = FP + 1
                if self.gt_image[i, j] >= 1 and self.pred_image[i, j] == 0:
                    FN = FN + 1

        if (FP+TN) == 0:
            fpr = 0
        else:
            fpr = float(FP)/float(FP+TN)

        if (TP+FN) == 0:
            tpr = 0
        else:
            tpr = float(TP)/float(TP+FN)

        return TP, TN, FP, FN, fpr, tpr


# Main method.
if __name__ == "__main__":

    gt_image = './Masks/Transversal/'
    pred_image = './Segmentation_Methods/KMeans Results/Transversal/'

    num_images = len(os.listdir(gt_image))

    y_true = [skimage.io.imread(file)
              for file in glob.glob(gt_image + "*.png")]

    pred = [skimage.io.imread(file)
            for file in glob.glob(pred_image + "*.png")]

    y_pred = []
    for i in range(len(pred)):
        new_pred = color.rgb2gray(pred[i])
        y_pred.append(new_pred)

    pa_array = []
    ma_array = []
    IOU_array = []
    mIOU_array = []
    fwIOU_array = []

    for i in range(num_images):
        val = Evaluate(y_pred[i], y_true[i])
        pa = val.pixel_accuracy()

        ma = val.mean_accuracy()
        IOU = val.IOU()
        mIOU = val.mean_IOU()
        fwIOU = val.fweight_IOU()

        pa_array.append(pa)
        ma_array.append(ma)
        IOU_array.append(IOU)
        mIOU_array.append(mIOU)
        fwIOU_array.append(fwIOU)

    # Compute the average of each metric.
    pa = np.mean(np.array(pa_array))
    ma = np.mean(np.array(ma_array))
    IOU = np.mean(np.array(IOU_array))
    mIOU = np.mean(np.array(mIOU_array))
    fwIOU = np.mean(np.array(fwIOU_array))

    print(" ")
    print("Average Pixel Accuracy: ", str(round(pa, 3)))
    print("Average Mean Accuracy: ", str(round(ma, 3)))
    print("Average IOU: ", str(round(IOU, 3)))
    print("Average Mean IOU: ", str(round(mIOU, 3)))
    print("Average Weighted IOU: ", str(round(fwIOU, 3)))

    # Write results to file.
    f = open('./Segmentation_Methods/'
             'Evaluation Results/KMeans/'
             'Transversal/KMeans.txt', 'w')

    f.write('Average Pixel Accuracy: %s \n' % (str(round(pa, 3))))
    f.write('Average Mean Accuracy: %s \n' % (str(round(ma, 3))))
    f.write('Average IOU: %s \n' % (str(round(IOU, 3))))
    f.write('Average Mean IOU: %s \n' % (str(round(mIOU, 3))))
    f.write('Average Weighted IOU: %s \n' % (str(round(fwIOU, 3))))

    f.close()
