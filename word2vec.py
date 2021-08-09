#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：使用word2vec将分词后的数据进行文本向量化，保存词语的索引字典、词向量，然后保存为pkl文件
word2vec：
	1.预训练语料：已分词的语料，输入格式为txt（一行为一条数据）或list of list
	2.建立一个空的模型对象；遍历一次语料库建立词典（统计每个词出现的次数，根据各词的词频建立哈夫曼树，得到词的哈夫曼编码）；
      第二次遍历语料库建立并训练模型（初始化词向量，逐句的读取一系列的词，用梯度下降法更新词向量）
	3.可保存的：模型，根据语料得到的索引字典{索引数字: 单词}，词向量（后两者通过pickle库保存）
	4.模型评估：词的向量表示，词与词之间的相似度，与某个词最相近的前N个词
"""

import pickle
import logging
import numpy as np

np.random.seed(1337)  # For Reproducibility
# from b_TextSta import TextSta   #自己定义的类，可获取列表形式的训练语料
import gensim
from gensim.models import Word2Vec, word2vec
from gensim.models import fasttext, FastText
from gensim.corpora.dictionary import Dictionary

file_path = "./model/word2vec.txt"
# 将日志输出到控制台
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# ==========================1.读取预训练语料=======================================
print("选择word2vec的已经分好词的文本...")
'''
path = addr + 'lemmas_sent.txt'
T = TextSta(path)
sentences = T.sen()#可获取列表形式的训练语料
'''
sentences = word2vec.LineSentence(file_path)

# =====================2.训练Word2vec模型（可尝试修改参数）...====================
'''
# FastText
print('训练FastText模型（可自定义参数）...')
model = FastText(sentences,
                 size=128,
                 min_count=1,
                 window=5)
model.wv.save('./model/fastText_128.vec')
model.wv.save_word2vec_format('./model/fastText_128.tsv', binary=False)

'''
print('训练Word2vec模型（可自定义参数）...')
model = Word2Vec(sentences,
                 size=128,  # 词向量维度
                 min_count=1,  # 词频阈值
                 window=5,  # 窗口大小
                 cbow_mean=1)

print(u"保存w2v模型...")
model.save('./model/words.model')  # 保存模型

model.wv.save_word2vec_format('./model/words.tsv', binary=False)  # 保存txt格式的词向量


# ===================3.创建词语字典，并返回word2vec模型中词语的索引，词向量================
def create_dictionaries(p_model):
    gensim_dict = Dictionary()  # 创建词语词典
    gensim_dict.doc2bow(p_model.wv.vocab.keys(), allow_update=True)
    w2dict_index = {v: k + 1 for k, v in gensim_dict.items()}  # 词语+索引。词语的索引，从1开始编号
    w2index_dict = {k + 1: v for k, v in gensim_dict.items()}  # 索引+词语。词语的索引，从1开始编号
    w2vec = {word: p_model.wv.get_vector(word) for word in w2dict_index.keys()}  # 词语的词向量
    return w2dict_index, w2index_dict, w2vec


# ====================4.从训练好的模型中提取出索引字典、词向量字典index_dict,==========================
dict_index, index_dict, word_vectors = create_dictionaries(model)

# ===========================5.使用 pickle 存储序列化数据 ====================================
# pickle是一个非常方便的库 可以将py的字典、列表等等程序运行过程中的对象存储为实体数据存储为pkl文件
print(u"保存word2vec_128.pkl文件...")
output = open("./model/word2vec_128.pkl", 'wb')
pickle.dump(dict_index, output)  # 索引字典，{单词: 索引数字}
pickle.dump(index_dict, output)  # 索引字典，{索引数字: 单词}
pickle.dump(word_vectors, output)  # 词向量字典
output.close()

if __name__ == "__main__":
    pass
