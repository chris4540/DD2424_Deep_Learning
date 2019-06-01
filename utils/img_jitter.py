"""
Make some a small random geometric and photometric jitter
    * flip
    * small rotation
    * color jitter

See also:
https://gombru.github.io/2017/09/14/data_augmentation/
https://blog.csdn.net/tsq292978891/article/details/79107775
https://www.jianshu.com/p/2078445c3ef7
"""

import numpy as np
from scipy import ndimage
from PIL import ImageEnhance
from PIL import Image
from tqdm import tqdm

class ImageJitter:

    def __init__(self):
        pass

    def jitter_batch(self, batchs):
        imgs =  self.batch_to_imgs(batchs)
        n_data = imgs.shape[0]
        ret = np.zeros_like(imgs)
        # for i in tqdm(range(n_data), desc="Jitter images"):
        for i in range(n_data):
            img = imgs[i, :]
            jittered_img = self.jitter_img(img)
            ret[i, :] = jittered_img

        ret_batch = self.imgs_to_batch(ret)
        return ret_batch

    def jitter_img(self, img):
        ret = np.array(img, copy=True)

        # random flip
        if np.random.choice([True, False]):
            ret = self.horrizonal_flip(ret)

        # random flip
        if np.random.choice([True, False]):
            rot_angle = np.random.uniform(-2, 2)
            ret = self.rotate(ret, rot_angle)

        # add noise
        if np.random.choice([True, False]):
            ret = self.add_gaussian_noisy(ret)
        else:
            ret = self.jitter_color(ret)
        return ret

    @staticmethod
    def batch_to_imgs(batch):
        """
        Args:
            batch (ndarray): batch.shape = (d, N)

        Return:
            imgs with dim (N, w, h, c) or (N, 32, 32, 3)
        """
        n_color = 3
        n_data = batch.shape[-1]
        n_dim = batch.shape[0]
        w = int(np.sqrt(n_dim // n_color))
        imgs = batch.reshape(n_color, w, w, n_data).transpose(3,1,2,0)
        return imgs

    @staticmethod
    def imgs_to_batch(imgs):
        """
        Reverse function of batch_to_imgs
        """
        n_data = imgs.shape[0]
        batch = imgs.transpose(3, 1, 2, 0).reshape(-1, n_data)
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
    def add_gaussian_noisy(img, mean=0.1, sigma=0.3):
        noisy = np.random.normal(mean, sigma, size=img.shape)
        ret = img + noisy
        return ret

    @staticmethod
    def jitter_color(img):
        """
        The impace factor are drawn from the uniform distribution
        factor = 1 will keep the orignal image
        factor = 0 will go to an extreme case
        """
        image = Image.fromarray(np.uint8(img*255))
        factor = np.random.uniform(0.8, 1)
        image = ImageEnhance.Color(image).enhance(factor)
        factor = np.random.uniform(0.8, 1)
        image = ImageEnhance.Brightness(image).enhance(factor)
        factor = np.random.uniform(0.8, 1)
        image = ImageEnhance.Contrast(image).enhance(factor)
        factor = np.random.uniform(0.8, 1.2)
        result_img = ImageEnhance.Sharpness(image).enhance(factor)
        ret = np.asarray(result_img) / 255
        return ret

if __name__ == "__main__":
    pass
    # check diff for batch_to_imgs and imgs_to_batch