import numpy as np

class TextLinesReader:

    def __init__(self, lines):
        self.lines = lines
        self._char_to_idx = dict()
        self._idx_to_char = dict()
        self.indexized_seq = list()
        self.char_set = set()

    def process(self):
        seq = list()
        for l in self.lines:
            chars_of_line = list(l)
            # extend it to indexized_seq
            seq.extend(chars_of_line)
            self.char_set.update(chars_of_line)

        # build the mapping
        for i, k in enumerate(self.char_set):
            self._char_to_idx[k] = i
            self._idx_to_char[i] = k

        # map the mapping from char to idx to the seq
        for c in seq:
            self.indexized_seq.append(self._char_to_idx[c])

    def get_seq(self):
        return self.indexized_seq

    def get_one_hot(self, part_seq):
        one_idx = np.array(part_seq)
        nkind = self.n_char
        nlabels = len(part_seq)
        ret = np.zeros((nkind, nlabels))
        ret[one_idx, np.arange(nlabels)] = 1
        return ret

    @property
    def n_char(self):
        """
        Getter of the number of unique character in the text file
        """
        ret = len(self.char_set)
        return ret

    # -------------------------------
    # Translator functions
    # -------------------------------
    def map_chars_to_idxs(self, chars):
        ret = list()
        default_idx = self._char_to_idx[' ']
        for c in chars:
            i = self._char_to_idx.get(c, default_idx)
            ret.append(i)
        return ret

    def map_idxs_to_chars(self, idxs):
        ret = list()
        default_char = ' '
        for i in idxs:
            c = self._idx_to_char.get(i, default_char)
            ret.append(c)
        return ret
