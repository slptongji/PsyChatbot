import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras.backend.tensorflow_backend as KTF
from sklearn.preprocessing import MultiLabelBinarizer
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense
from keras.layers import GRU, Bidirectional
from keras import backend as K
from keras.engine.topology import Layer
from tqdm import tqdm
from albert_zh.extract_feature import BertVector


class Attention(Layer):
    def __init__(self, attention_size, **kwargs):
        self.attention_size = attention_size
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # W: (EMBED_SIZE, ATTENTION_SIZE)
        # b: (ATTENTION_SIZE, 1)
        # u: (ATTENTION_SIZE, 1)
        self.W = self.add_weight(name="W_{:s}".format(self.name),
                                 shape=(input_shape[-1], self.attention_size),
                                 initializer="glorot_normal",
                                 trainable=True)
        self.b = self.add_weight(name="b_{:s}".format(self.name),
                                 shape=(input_shape[1], 1),
                                 initializer="zeros",
                                 trainable=True)
        self.u = self.add_weight(name="u_{:s}".format(self.name),
                                 shape=(self.attention_size, 1),
                                 initializer="glorot_normal",
                                 trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x, mask=None):
        # input: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        # et: (BATCH_SIZE, MAX_TIMESTEPS, ATTENTION_SIZE)
        et = K.tanh(K.dot(x, self.W) + self.b)
        # at: (BATCH_SIZE, MAX_TIMESTEPS)
        at = K.softmax(K.squeeze(K.dot(et, self.u), axis=-1))
        if mask is not None:
            at *= K.cast(mask, K.floatx())
        # ot: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        atx = K.expand_dims(at, axis=-1)
        ot = atx * x
        # output: (BATCH_SIZE, EMBED_SIZE)
        output = K.sum(ot, axis=1)
        return output

    def compute_mask(self, input, input_mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        config = {"attention_size": self.attention_size}
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == "__main__":
    config = tf.ConfigProto()  
    config.gpu_options.allow_growth=True   
    sess = tf.Session(config=config)
    KTF.set_session(sess)  

    # load data
    label_set = []
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    with open("./data/train.txt", "r", encoding="utf-8") as f:
        train_set = [_.strip() for _ in f.readlines()]
    with open("./data/test.txt", "r", encoding="utf-8") as f:
        test_set = [_.strip() for _ in f.readlines()]

    for line in test_set:
        labels = line.split(" ", maxsplit=1)[0].split(",")
        label_set.append(labels)
    mlb = MultiLabelBinarizer()
    mlb.fit(label_set)
    with open("./data/topic_type.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(mlb.classes_.tolist(), ensure_ascii=False, indent=4))
    print("Total number of topics: %d" % len(mlb.classes_))

    for line in train_set:
        labels = line.split(" ", maxsplit=1)[0].split(",")
        y_train.append(mlb.transform([labels])[0])

    for line in test_set:
        labels = line.split(" ", maxsplit=1)[0].split(",")
        y_test.append(mlb.transform([labels])[0])

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    print('y_train.shape: %s' % str(y_train.shape))
    print('y_test.shape: %s' % str(y_test.shape))

    # encoding
    print("Encoding...")
    bert_model = BertVector(pooling_strategy="NONE", max_seq_len=200)
    f = lambda text: bert_model.encode([text])["encodes"][0]

    for ex in tqdm(train_set, 'Encoding'):
        desc = ex.split(" ", maxsplit=1)[1]
        x_train.append(f(desc)) 

    for ex in tqdm(test_set, 'Encoding'):
        desc = ex.split(" ", maxsplit=1)[1]
        x_test.append(f(desc))

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    print('x_train.shape: '+str(x_train.shape))
    print('x_test.shape: '+str(x_test.shape))

    # model training
    inputs = Input(shape=(200, 312, ), name="input")
    gru = Bidirectional(GRU(128, dropout=0.5, return_sequences=True), name="bi-gru")(inputs)
    attention = Attention(32, name="attention")(gru)
    num_class = len(mlb.classes_)
    output = Dense(num_class, activation='sigmoid', name="dense")(attention)
    model = Model(inputs, output)

    model.compile(loss='binary_crossentropy',
                optimizer=Adam(),
                metrics=['accuracy'])

    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=128, epochs=10)
    model.save('./output/model.h5')

    # visualizing
    plt.subplot(2, 1, 1)
    epochs = len(history.history['loss'])
    plt.plot(range(epochs), history.history['loss'], label='loss')
    plt.plot(range(epochs), history.history['val_loss'], label='val_loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    epochs = len(history.history['accuracy'])
    plt.plot(range(epochs), history.history['accuracy'], label='acc')
    plt.plot(range(epochs), history.history['val_accuracy'], label='val_acc')
    plt.legend()
    plt.savefig("./output/loss_acc.png")
