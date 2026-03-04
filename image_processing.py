import cv2
import numpy as np
import torch
def preprocessing_images_for_prediction(image, datset_std): #std (computed once over dataset)
    # img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) 
    img = cv2.resize(image, (211,189), interpolation=cv2.INTER_LINEAR)
    img = 255 - img # image inversion 
    img = img.astype(np.float32) # required for division

    img = img/datset_std #normalize each image by dividing the pixel values with the standard deviation of the pixel

    # When we load a grayscale image:(h,w) and convolution layer expects (channels, h, w)
    #so for the single gray scale (h,w), img = np.expand_dims(img, axis=0) it inserts a new dimention at position 0 -> (1, 155, 220)

    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)

    return torch.tensor(img, dtype=torch.float32)


def calculate_std_images(dataset):
    all_pixel = []

    for p_image in dataset:
        # image = cv2.imread(p_image)
        
        all_pixel.append(np.asarray(p_image).flatten())
        

    all_pixel = np.concatenate(all_pixel)

    std = np.std(all_pixel)
    return std

