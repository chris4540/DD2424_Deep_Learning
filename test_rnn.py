
class TextLinesReader:

    def __init__(self, lines):
        self.lines = lines
        self._char_to_idx = dict()
        self._idx_to_char = dict()
        self.indexized_seq = list()
        self.char_set = set()

    def process(self):
        seq = list()
        for l in lines:
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


def ReadAndProcess(datafile):
    """ Read the .txt file and return two dicts `char_to_ind` and `ind_to_char`. """
    with open(datafile, 'r') as f:
        l = f.read()
    char = list(set(l))
    K = len(char)
    ind = [i for i in range(len(char))]
    char_to_ind = dict(zip(char, ind))
    ind_to_char = dict(zip(ind ,char))
    return char_to_ind, ind_to_char, K, l

if __name__ == "__main__":
    with open("rnn_data/goblet_book.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # print(lines[0])
    # print(list(lines[0]))
    reader = TextLinesReader(lines)
    reader.process()
    print(reader.n_char)
    print(reader._char_to_idx)
    print(reader._idx_to_char)

    string = "chris 4540"
    idxs = reader.map_chars_to_idxs(list(string))
    print(idxs)
    chars = reader.map_idxs_to_chars(idxs)
    print(chars)

    # char_to_ind, ind_to_char, K, l = ReadAndProcess("rnn_data/goblet_book.txt")
    # print(char_to_ind)
    # for l in lines:
    #     print(l)
    #     break

