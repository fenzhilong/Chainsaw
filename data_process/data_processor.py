# coding=utf-8
import re
from collections import Counter
import tensorflow as tf
from tensorflow import keras
import pyjieba as jieba
from tqdm import tqdm
from utils.path_helper import path_chainsaw

lines_train = open(path_chainsaw + "/data/cnews/cnews.train.txt", encoding="UTF-8").readlines()
lines_val = open(path_chainsaw + "/data/cnews/cnews.val.txt", encoding="UTF-8").readlines()
lines_test = open(path_chainsaw + "/data/cnews/cnews.test.txt", encoding="UTF-8").readlines()

stopwords = {word.strip() for word in open(path_chainsaw + "/data/stopwords.txt", encoding="UTF-8").readlines()}


def vocab(vocab_size=40000):
    vocab_list = list()
    for lines_data in [lines_val, lines_test, lines_train]:
        for line in tqdm(lines_data):
            text_and_label = line.split("\t")
            # text = re.split(r"\s+", text_and_label[0].strip())
            text = jieba.cut(text_and_label[1].strip())
            # label = re.split(r"__", text_and_label[1].strip())
            for word in text:
                if re.match(r"[_0-9]+", word) or word in stopwords:
                    continue
                vocab_list.append(word)

    temp_vocab = Counter(vocab_list).most_common(vocab_size)
    final_vocab = list()
    with open(path_chainsaw + "/data/cnews/cnews.vocab_word.txt", "a", encoding='UTF-8') as f:
        for word, count in temp_vocab:
            final_vocab.append(word)
            f.write(word + "\n")

    return final_vocab


vocab_list = [word.strip() for word in
              open(path_chainsaw +"/data/cnews/cnews.vocab_word.txt", encoding="UTF-8").readlines()]

labels = ["体育", "娱乐", "家居", "彩票", "房产", "教育", "时尚", "时政", "星座", "游戏", "社会", "科技", "股票", "财经"]
index2label = {index: label for index, label in enumerate(labels)}
label2index = {label: index for index, label in index2label.items()}


def load_text_data(lines_data, vocab_size=20000, sentence_len=30, doc_len=20, max_len=600, han=False):
    index2word = {0: "padding", 1: "unknown"}
    if len(vocab_list) > vocab_size:
        final_vocab = vocab_list[:vocab_size]
    else:
        final_vocab = vocab_list
    _index2word = {i + len(index2word): word for i, word in enumerate(final_vocab)}
    index2word.update(_index2word)
    word2index = {word: index for index, word in index2word.items()}
    X, Y = [], []
    for line in tqdm(lines_data):
        text_and_label = line.split("\t")
        try:
            text = jieba.cut(text_and_label[1].strip())
            label = text_and_label[0].strip()
        except Exception as e:
            print(e)
            continue
        x = []
        for word in text:
            if word in word2index:
                x.append(word2index[word])
            else:
                x.append(word2index["unknown"])
        X.append(x)
        Y.append(label2index[label])

    if han is True:
        X = keras.preprocessing.sequence.pad_sequences(X, padding='post', maxlen=sentence_len * doc_len, dtype='int32')
        X = tf.reshape(X, shape=[-1, doc_len, sentence_len])
    else:
        X = keras.preprocessing.sequence.pad_sequences(X, padding='post', maxlen=max_len, dtype='int32')

    return tf.convert_to_tensor(X), tf.one_hot(indices=Y, depth=len(label2index))
