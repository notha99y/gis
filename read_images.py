import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal
from pathlib import Path

class EMSpectrum:
    def __init__(self):
        self.blue = 450 #nm
        self.green = 500 #nm
        self.red = 750 #nm

class Sentinel2_MSI:
    def __init__(self):
        self.B01 = 443 # Aerosols
        self.B02 = 490 # Blue
        self.B03 = 560 # Green
        self.B04 = 665 # Red
        self.B05 = 705 # Red edge 1
        self.B06 = 740 # Red edge 2
        self.B07 = 783 # Red edge 3
        self.B08 = 842 # NIR
        self.B08A = 865 # Red edge 4
        self.B09 = 945 # Water vapor
        self.B10 = 1375 # Cirrus
        self.B11 = 1610 # SWIR 1
        self.B12 = 2190 # SWIR 2
    
    def read_image(self, image_path):
        '''Reads a Sentinel-2's MSI path 
        '''
        self.image_name = image_path.stem
        self.raster = gdal.Open(str(image_path))
        self.rasterArray12 = self.raster.ReadAsArray()
        self.rasterArray8 = (self.rasterArray12/16).astype('uint8')
        print('Image Read. Dimension of array: ', self.rasterArray12.shape)

    def display_RGB(self, ax=None):
        '''Shows an image in RGB
        '''

        R = np.expand_dims(self.rasterArray8[3], axis = -1)
        G = np.expand_dims(self.rasterArray8[2], axis = -1)
        B = np.expand_dims(self.rasterArray8[1], axis = -1)
        RGB_image = np.concatenate((R,G,B), axis=-1)
        if ax:
            ax.set_title('RGB Image')
            ax.set_axis_off()
            ax.imshow(RGB_image)
            
        else:
            plt.title('RGB Image')
            plt.imshow(RGB_image)
            
    def display_NIR(self, ax=None):
        N = self.rasterArray8[7]
        if ax:
            ax.imshow(N)
            ax.set_title('NIR')
            ax.set_axis_off()
            
        else:
            plt.imshow(N)
            plt.title('NIR')
    
    def display_Aerosols(self, ax=None):
        A = self.rasterArray8[0]
        if ax:
            ax.imshow(A)
            ax.set_title('Aerosols')
            ax.set_axis_off()
            
        else:
            plt.imshow(A)
            plt.title('Aerosols')
            

    def display_WaterVapor(self, ax=None):
        W = self.rasterArray8[9]
        if ax:
            ax.imshow(W)
            ax.set_title('Water Vapor')
            ax.set_axis_off()
            
        else:
            plt.imshow(W)
            plt.title('Water Vapor')
            
    def display_Cirrus(self, ax=None):
        C = self.rasterArray8[10]
        
        if ax:
            ax.imshow(C)
            ax.set_title('Cirrus')
            ax.set_axis_off()
            
        else:
            plt.imshow(C)
            plt.title('Cirrus')
            
    def display_SWIR1(self, ax=None):
        S1 = self.rasterArray8[11]
        if ax:
            ax.imshow(S1)
            ax.set_title('SWIR1')
            ax.set_axis_off()
            
        else:
            plt.imshow(S1)
            plt.title('SWIR1')
            
    def display_SWIR2(self, ax=None):
        S2 = self.rasterArray8[12]
        if ax:
            ax.imshow(S2)
            ax.set_title('SWIR2')
            ax.set_axis_off()
            
        else:
            plt.imshow(S2)
            plt.title('SWIR2')
            
    def display_them(self):
        fig, ax = plt.subplots(2,4, figsize = (12,12))
        # fig.title(self.image_name)
        self.display_RGB(ax=ax[0,0])
        self.display_Aerosols(ax=ax[0,1])
        self.display_Cirrus(ax=ax[0,2])
        self.display_WaterVapor(ax=ax[1,0])
        self.display_NIR(ax=ax[0,3])
        self.display_SWIR1(ax=ax[1,1])
        self.display_SWIR2(ax=ax[1,2])

        
if __name__ == "__main__":
    paths = list(Path('AnnualCrop').rglob('*.tif'))
    test_path = paths[0]
    print(test_path)
    msi = Sentinel2_MSI()
    msi.read_image(test_path)
    msi.display_RGB()
    msi.display_Aerosols()
    msi.display_Cirrus()
    msi.display_WaterVapor()
    msi.display_SWIR1()
    msi.display_SWIR2()
    msi.display_them()
    plt.show()
    # print(type(raster))
    # rasterArray = raster.ReadAsArray()
    # print(type(rasterArray))
    # print(rasterArray.shape)
    # img = cv2.imread(str(test_path))
    # img = open_image(str(test_path))
    # cv2.imshow('satellite img', img)
