"""
Make some a small random geometric and photometric jitter
    * flip
    * small rotation
    * color jitter

See also:
https://gombru.github.io/2017/09/14/data_augmentation/
https://blog.csdn.net/tsq292978891/article/details/79107775
"""
import numpy as np
from scipy import ndimage
from PIL import ImageEnhance
from PIL import Image

class ImageArgument:

    def __init__(self):
        pass

    def jitter(self, batchs):
        pass

    @staticmethod
    def batch_to_imgs(batch):
        """
        Args:
            batch (ndarray): batch.shape = (d, N)

        Return:
            imgs with dim (N, w, h, c) or (N, 32, 32, 3)
        """
        n_color = 3
        n_batch = batch.shape[-1]
        n_dim = batch.shape[0]
        w = int(np.sqrt(n_dim // n_color))
        imgs = batch.reshape(n_color, w, w, n_batch).transpose(3,1,2,0)
        return imgs

    @staticmethod
    def imgs_to_batch(imgs):
        """
        Reverse function of batch_to_imgs
        """
        n_batch = imgs.shape[0]
        batch = imgs.transpose(3, 1, 2, 0).reshape(-1, n_batch)
        return batch

    @staticmethod
    def horrizonal_flip(img):
        """
        horrizonal flip the input img
        """
        # flip the second axis
        flipped = img[:, ::-1, :]
        return flipped

    @staticmethod
    def rotate(img, angle):
        """
        Args:
            img (ndarray): img.shape = (w, h, c)
            angle (float): the rotation angle, expected to be small
        """
        rotated = ndimage.rotate(img, angle, reshape=False, mode='nearest')
        assert rotated.shape == img.shape
        return rotated

    @staticmethod
    def add_gaussian_noisy(img, mean=0.05, sigma=0.03):
        noisy = np.random.normal(mean, sigma, size=img.shape)
        ret = img + noisy
        return ret

    @staticmethod
    def jitter_color(img):
        """
        """
        image = Image.fromarray(np.uint8(img*255))
        random_factor = np.random.randint(5, 10) / 10.
        color_image = ImageEnhance.Color(image).enhance(random_factor)
        random_factor = np.random.randint(10, 21) / 10.
        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)
        random_factor = np.random.randint(10, 21) / 10.
        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)
        random_factor = np.random.randint(7, 10) / 10.
        result_img = ImageEnhance.Sharpness(contrast_image).enhance(random_factor)
        ret = np.asarray(result_img) / 255
        return ret

if __name__ == "__main__":
    pass
    # check diff for batch_to_imgs and imgs_to_batch