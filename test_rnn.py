from utils.text_preproc import TextLinesReader
from clsr.rnn import VanillaRNN

if __name__ == "__main__":
    # with open("rnn_data/goblet_book.txt", 'r', encoding='utf-8') as f:
    #     lines = f.readlines()

    # reader = TextLinesReader(lines)
    # reader.process()

    rnn = VanillaRNN()
    pass