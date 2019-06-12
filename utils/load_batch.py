import pickle
import numpy as np

def unpickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding='bytes')

    # convert the key from python2 string to python3 string
    ret = dict()
    for k in data.keys():
        ret[k.decode("utf-8")] = data[k]
    return ret

def load_batch(filename, dtype='float32'):
    """
    Read the batch file and transform the data to double

    Args:
        the file
        dtype (str): the type of the converted data
    Return:
        return a dictionary
    """
    data = unpickle(filename)

    float_pixel = data["data"].T / 255
    float_pixel = float_pixel.astype(dtype)

    # build the return
    ret = {
        "pixel_data": float_pixel,
        # "onehot_labels": onehot_labels,
        "labels": np.array(data["labels"])
    }

    return ret

def merge_batch(list_of_filenames):
    ret = None
    for f in list_of_filenames:
        tmp = load_batch(f)
        if ret is None:
            ret = tmp
        else:
            for k in ret.keys():
                ret[k] = np.hstack((ret[k], tmp[k]))

    return ret

class cifar10_DataLoader:

    def __init__(self, data, batch_size=100, shuffle=False):
        """
        Generator of batched data
        Args:
            data (dict):
                data['pixel_data'] is all input data and
                data['labels'] is all labels for the data
            batch_size (int): batch size

        Return:
            tuple of input and labels

        Example:
            >>> for inputs, labels in dataloader(data, batch_size=100):
                    # make use of inputs and labels
                    pass
        """
        self.all_labels = data['labels']
        self.all_inputs = data['pixel_data']
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.idx =  np.arange(self.all_labels.shape[0])

    def _shuffle_idx(self):
        np.random.shuffle(self.idx)

    def iterator(self):
        if self.shuffle:
            self._shuffle_idx()
        n_data = self.all_labels.shape[0]
        for j in range(int(np.ceil(n_data / self.batch_size))):
            j_s = j * self.batch_size
            j_e = (j+1) * self.batch_size
            indices = self.idx[j_s:j_e]
            batch_inputs = self.all_inputs[:, indices]
            batch_labels = self.all_labels[indices]
            yield batch_inputs, batch_labels

    def __iter__(self):
        return self.iterator()
