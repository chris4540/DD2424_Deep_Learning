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

# def get_label_to_one_hot(labels):
#     """
#     Create a one hot encoding matrix from labels

#     Args:
#         labels (list[int])

#     Return:
#         one hot encoding matrix of the labels
#     """
#     one_idx = np.array(labels)
#     nkind = len(np.unique(one_idx))
#     nlabels = len(one_idx)

#     ret = np.zeros((nkind, nlabels))
#     ret[one_idx, np.arange(nlabels)] = 1

#     return ret

def load_batch(filename):
    """
    Read the batch file and transform the data to double

    Args:
        the file

    Return:
        return a dictionary
    """
    data = unpickle(filename)

    float_pixel = data["data"].T / 255
    # onehot_labels = get_label_to_one_hot(data["labels"])
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
