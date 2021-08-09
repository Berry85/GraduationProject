# -*- coding: utf-8 -*-
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Embedding, LSTM
from keras.layers import Dropout, Activation, Flatten, concatenate
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import gc
import tensorflow as tf
import pickle
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tf.compat.v1.disable_eager_execution()

X_FILE_PATH = "./Data/Event.csv"
AFTER_FILE_PATH = "./Data/After.csv"
CONCURRENT_FILE_PATH = "./Data/Concurrent.csv"
Y_FILE_PATH = "./Data/Output.csv"

EMBEDDING_FILE_PATH = "./model/words.tsv"
OUTPUT_PATH = "./Output/model.h5"
TOKENIZER_PATH = "./Output/tokenizer.pickle"
PREDICTED_PATH = "./Output/predicted.csv"
IMG_PATH = "./Output/Img/Img.png"

#
BATCH_SIZE = 32
EMBEDDING_DIM = 128
MAX_NB_WORDS = 20000


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(" — val_f1: % f — val_precision: % f — val_recall % f" % (_val_f1, _val_precision, _val_recall))
        return


def load_file():
    # 读取文件
    x = pd.read_csv(X_FILE_PATH)
    after = pd.read_csv(AFTER_FILE_PATH, sep=" ")
    concurrent = pd.read_csv(CONCURRENT_FILE_PATH, sep=" ")
    x.drop(columns=["id"], inplace=True)
    after.drop(columns=["id"], inplace=True)
    concurrent.drop(columns=["id"], inplace=True)

    # 转换格式类型
    x['Age'] = x['Age'].apply(str)
    x['ConditionAge'] = x['ConditionAge'].apply(str)
    x['DuringTime'] = x['DuringTime'].apply(str)
    #
    after = after['after_type'].map(lambda x: x.split(' '))
    concurrent = concurrent['concurrent_type'].map(lambda x: x.split(' '))

    # 转换至ndarray
    x = x.values
    after = after.values
    concurrent = concurrent.values

    # after输入
    after_encoder = MultiLabelBinarizer()
    after_encoder.fit(after)
    encoder_after = after_encoder.fit_transform(after)
    after = encoder_after
    # concurrent输入
    concurrent_encoder = MultiLabelBinarizer()
    concurrent_encoder.fit(concurrent)
    encoder_concurrent = concurrent_encoder.fit_transform(concurrent)
    concurrent = encoder_concurrent

    print("-----------------------")
    print("load_shape:")
    print("x:", x.shape)
    print("after:", after.shape)
    print("concurrent:", concurrent.shape)

    return x, after, after_encoder, concurrent, concurrent_encoder


def load_embedding(x):
    f = open(EMBEDDING_FILE_PATH, 'r', encoding='utf8')
    embedding_index = dict()
    for line in f:
        values = line.split()
        word = str(values[0])
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs
    f.close()
    print("-----------------------")
    print('Load %s word vectors.' % len(embedding_index))

    # 因为需要增加未登录的词，所有+1
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS + 1, lower=False)

    # x输入
    x_doc = []
    for i in range(0, len(x)):
        x_doc.append(list(x[i]))

    # 读取训练资料
    tokenizer.fit_on_texts(x_doc)
    word_index = tokenizer.word_index
    print("-----------------------")
    print('Found %s unique tokens.' % len(word_index))
    x_doc = tokenizer.texts_to_sequences(x_doc)

    # 把输出对齐
    x_doc = pad_sequences(x_doc, padding='post')

    num_words = min(MAX_NB_WORDS, len(word_index)) + 1
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    gc.collect()
    embeddedCount = 0

    for word, i in word_index.items():
        i -= 1
        embedding_vector = embedding_index.get(str(word))
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            embeddedCount += 1
        elif embedding_vector is None:
            pass
            # print("word:", word)
            # print("i:", i)
            # print("vec:", embedding_vector)
    print("-----------------------")
    print('total_embedded:', embeddedCount, 'commen words')
    del embedding_index
    gc.collect()
    # 把tokenizer保存
    with open(TOKENIZER_PATH, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("-----------------------")
    print('Tokenizer已经保存到Output文件夹')
    return embedding_matrix, tokenizer, x_doc, x


# 拆分训练数据
def split_test(train_X, after_train, concurrent_train):
    # 打乱数据集

    index = np.arange(len(train_X))
    np.random.shuffle(index)
    train_X = train_X[index]
    after_train = after_train[index]
    concurrent_train = concurrent_train[index]

    train_X = np.asarray(train_X)
    after_train = np.asarray(after_train)
    concurrent_train = np.asarray(concurrent_train)

    print(train_X.shape, after_train.shape, concurrent_train.shape)
    shape_x = int(train_X.shape[0] * 0.8)
    shape_y = int(after_train.shape[0] * 0.8)

    # 分train和test
    x_train, x_test = train_X[:shape_x, :], train_X[shape_x:, :]
    after_train, after_test = after_train[:shape_y, :], after_train[shape_y:, :]
    concurrent_train, concurrent_test = concurrent_train[:shape_y, :], concurrent_train[shape_y:, :]

    print("-----------------------")
    print("x_train.shape : ", x_train.shape)
    print("x_test.shape : ", x_test.shape)
    print("after_train.shape : ", after_train.shape)
    print("after_test.shape : ", after_test.shape)
    print("concurrent_train.shape : ", concurrent_train.shape)
    print("concurrent_test.shape : ", concurrent_test.shape)
    return x_train, x_test, after_train, after_test, concurrent_train, concurrent_test


def build_model(embedding_matrix, tokenizer,
                x_train, x_test,
                after_train, after_test, after_encoder,
                concurrent_train, concurrent_test, concurrent_encoder):
    # model = Sequential()
    main_input = Input(shape=(10,),
                       name="main_input")

    x = Embedding(input_dim=len(tokenizer.word_index) + 1,
                  output_dim=embedding_matrix.shape[1],
                  weights=[embedding_matrix],
                  trainable=True)(main_input)
    x = LSTM(units=EMBEDDING_DIM,
             return_sequences=False,
             name="lstm")(x)
    x = Dense(64,
              activation="relu",
              name="dense")(x)
    x = Dropout(0.5,
                name="dropout")(x)
    after_output = Dense(after_train.shape[1],
                         input_dim=EMBEDDING_DIM,
                         activation='softmax',
                         name="after_output")(x)
    concurrent_output = Dense(concurrent_train.shape[1],
                              input_dim=EMBEDDING_DIM,
                              activation='softmax',
                              name="concurrent_output")(x)

    model = Model(input=main_input, outputs=[after_output, concurrent_output])
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model.summary()
    plot_model(model, to_file=IMG_PATH, show_shapes=True)
    history = model.fit(x_train, [after_train, concurrent_train],
                        verbose=2,
                        batch_size=BATCH_SIZE,
                        epochs=10)

    score = model.evaluate(x_test, [after_test, concurrent_test], batch_size=BATCH_SIZE, verbose=2)
    print("-----------------history---------------")
    f = open("./Output/output.txt", mode="w+")
    f.write("-----------------------------------\n")
    f.write("loss: " + " ".join('%s' % id for id in history.history['loss']) + "\n")
    f.write("after_output_loss: " + " ".join('%s' % id for id in history.history['after_output_loss']) + "\n")
    f.write("concurrent_output_loss: " + " ".join('%s' % id for id in history.history['concurrent_output_loss']) + "\n")
    f.write("after_output_acc: " + " ".join('%s' % id for id in history.history['after_output_acc']) + "\n")
    f.write("concurrent_output_acc: " + " ".join('%s' % id for id in history.history['concurrent_output_acc']) + "\n")
    f.write("-----------------------------------\n")
    f.write("score" + " ".join('%s' % id for id in score) + "\n")
    f.close()
    print("-----------------score---------------")
    print('loss:', score[0])
    print('after_loss:', score[1])
    print('concurrent_loss:', score[2])
    print('after_accuracy:', score[3] * 100)
    print('concurrent_accuracy:', score[4] * 100)

    plt.plot(history.history['after_output_acc'])
    plt.title('After accuracy:')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.savefig("./Output/Img/After_acc.png")
    plt.show()

    plt.plot(history.history['concurrent_output_acc'])
    plt.title('Concurrent accuracy:')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.savefig("./Output/Img/Concurrent_acc.png")
    plt.show()

    plt.plot(history.history['loss'])
    plt.title('Model loss:')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig("./Output/Img/loss.png")
    plt.show()

    predicted = model.predict(x_test, verbose=2)
    after, concurrent = predicted[0], predicted[1]

    for i in range(len(after)):
        max_value = max(after[i])
        for j in range(len(after[i])):
            if max_value == after[i][j]:
                after[i][j] = 1
            else:
                after[i][j] = 0

    for i in range(len(concurrent)):
        max_value = max(concurrent[i])
        for j in range(len(concurrent[i])):
            if max_value == concurrent[i][j]:
                concurrent[i][j] = 1
            else:
                concurrent[i][j] = 0

    after = np.reshape(after, newshape=(-1))
    concurrent = np.reshape(concurrent, newshape=(-1))
    after_test = np.reshape(after_test, newshape=(-1))
    concurrent_test = np.reshape(concurrent_test, newshape=(-1))

    f1_after = f1_score(after, after_test)
    f1_concurrent = f1_score(concurrent, concurrent_test)
    precision_after = precision_score(after, after_test)
    precision_concurrent = precision_score(concurrent, concurrent_test)
    recall_after = recall_score(after, after_test)
    recall_concurrent = recall_score(concurrent, concurrent_test)
    print("f1_after:", f1_after)
    print("f1_concurrent:", f1_concurrent)
    print("precision_after:", precision_after)
    print("precision_concurrent:", precision_concurrent)
    print("recall_after:", recall_after)
    print("recall_concurrent:", recall_concurrent)
    print("=========================")

    pre_after = after_encoder.inverse_transform(predicted[0])
    pre_concurrent = concurrent_encoder.inverse_transform(predicted[1])

    x_test = tokenizer.sequences_to_texts(x_test)
    x_test = pd.DataFrame(x_test, dtype=str)
    pre_after = pd.DataFrame(pre_after, dtype=str)
    pre_concurrent = pd.DataFrame(pre_concurrent, dtype=str)
    # output = pd.concat([x_test, pre_after, pre_concurrent], axis=1, join="inner")
    pre_after.to_csv("./Output/pre_after.csv", index=False, sep=" ")
    pre_concurrent.to_csv("./Output/pre_concurrent.csv", index=False, sep=" ")
    output = pd.DataFrame(x_test)

    output.to_csv(PREDICTED_PATH, index=False, sep=" ")

    print("预测已输入至" + PREDICTED_PATH)
    print("------------------------")
    model.save(OUTPUT_PATH)
    print("model保存至：" + OUTPUT_PATH)


if __name__ == '__main__':
    x_train, after_train, after_encoder, concurrent_train, concurrent_encoder = load_file()
    embedding_matrix, tokenizer, x_train, x = load_embedding(x_train)

    x_train, x_test, after_train, after_test, concurrent_train, concurrent_test = split_test(
        x_train,
        after_train,
        concurrent_train)

    build_model(embedding_matrix, tokenizer,
                x_train, x_test,
                after_train, after_test, after_encoder,
                concurrent_train, concurrent_test, concurrent_encoder)
