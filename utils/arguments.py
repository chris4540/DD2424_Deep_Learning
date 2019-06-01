"""
Make some a small random geometric and photometric jitter
    * flip
    * small rotation
    * color jitter

See also:
https://gombru.github.io/2017/09/14/data_augmentation/
"""
import numpy as np

class ImageArgument:

    def __init__(self):
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



if __name__ == "__main__":
    pass
    # check diff for batch_to_imgs and imgs_to_batch