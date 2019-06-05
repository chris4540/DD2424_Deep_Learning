# -*- coding: utf-8 -*-
import zipfile
import json
from pathlib import Path
import re

def read_json_zip(path):
    ret = list()
    with zipfile.ZipFile(path, "r") as z:
        for filename in z.namelist():
            with z.open(filename) as f:
                data = f.read()
                ret.extend(json.loads(data))
    return ret

class TextFilter:

    def __init__(self):
        pass

    def do_filter(self, text):
        ret = text
        ret = self.remove_weblink(ret)
        ret = self.remove_ref_user(ret)
        ret = self.remove_hash_tag(ret)
        ret = self.remove_non_ascii_char(ret)
        return ret

    def remove_weblink(self, text):
        ret = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
        return ret

    def remove_ref_user(self, text):
        """
        remove @ symbol
        """
        ret = re.sub(r'@', '', text)
        return ret

    def remove_hash_tag(self, text):
        ret = re.sub(r'#', '', text)
        return ret

    def remove_non_ascii_char(self, text):
        ret = re.sub(r'[^\x00-\x7F]+',' ', text)
        return ret



if __name__ == "__main__":
    all_tweets = list()
    p = Path('rnn_data/trump_tweets/')
    for json_zip in p.glob('condensed_*.json.zip'):
        all_tweets.extend(read_json_zip(json_zip))

    texts = []
    for tw in all_tweets:
        texts.append(tw['text'])


    filter_ = TextFilter()
    with open('trump_tweets.txt', 'w', encoding="utf-8") as f:
        for t in texts:
            filtered_txt = filter_.do_filter(t)
            if len(filtered_txt) > 0:
                f.write("%s\n" % filtered_txt)
    # filter_ = TextFilter()
    # text = """
    # Trump International Tower http://bit.ly/sqvQq
    # """
    # print(filter_.remove_weblink(text))

    # text = "Tonight I trade places with Larry King @kingsthings and ..."
    # print(filter_.remove_ref_user(text))

    # text = "Congratulations to #TeamUSA🇺🇸🏆on your gr"
    # print(filter_.remove_hash_tag(text))
    # text = "Congratulations to #TeamUSA🇺🇸🏆on your gr"
    # print(filter_.remove_non_ascii_char(text))


