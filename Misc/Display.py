# Import the required libraries.
import matplotlib.pyplot as plt
from nilearn import plotting


# Create the Image Display class.
class ImageDisplay(object):
    # Class Constructor.
    def __init__(self, ImageDirectory=" ", MaskDirectory=" "):
        self.imageDir = ImageDirectory
        self.maskDir = MaskDirectory

    # Basic anatomical plot of the three image planes.
    def anatomicalPlot(self, image):
        plotting.plot_anat(
                image,
                cut_coords=[0, 0, 0],
                draw_cross=False,
                annotate=False
        )
        plotting.plot_anat(
                image,
                cut_coords=(-10, 0, 0),
                draw_cross=False,
                annotate=False
            )
        plotting.show()

    # Overlay the image mask with the original image.
    def overlayMask(self, image, mask):
        plotting.plot_roi(
                mask,
                image,
                cut_coords=[0, 0, 0],
                annotate=False,
                draw_cross=False
            )
        plotting.show()

    # Create the function to display a single image.
    def displayImages(self, dataFile, numImages, startSlice, incSlices):
        fig, ax = plt.subplots(1, numImages, figsize=[18, 6])
        self.img = self.image.loadImage(self.imageDir + dataFile)
        self.imageData = self.img.get_fdata()

        # Loop through the parameter range.
        for i in range(numImages):
            ax[i].imshow(self.imageData[startSlice, :, :], 'gray')
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            ax[i].set_title('Slice number: {}'.format(startSlice), color='r')
            startSlice += incSlices

        # Show the images.
        fig.subplots_adjust(wspace=0, hspace=0)
        plt.show()

    # Copy of previous function but intakes a resampled image object.
    def displayResampledImage(self, image, numImages, startSlice, incSlices):
        fig, ax = plt.subplots(1, numImages, figsize=[18, 6])

        # Loop through the parameter range.
        for i in range(numImages):
            ax[i].imshow(image[startSlice, :, :], 'gray')
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            ax[i].set_title('Slice number: {}'.format(startSlice), color='r')
            startSlice += incSlices

        # Show the images.
        fig.subplots_adjust(wspace=0, hspace=0)
        plt.show()
