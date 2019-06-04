from utils.text_preproc import TextLinesReader

if __name__ == "__main__":
    with open("rnn_data/goblet_book.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()

    reader = TextLinesReader(lines)
    reader.process()

    string = "chris 4540"
    idxs = reader.map_chars_to_idxs(list(string))
    print(idxs)
    chars = reader.map_idxs_to_chars(idxs)
    print(chars)