import numpy as np
import pandas as pd

df = pd.read_csv("../../data/improved_dataset_v2.csv")
df.head()

words = list(set(df["words"].values))
words.append("ENDPAD")
n_words = len(words)

tags = list(set(df["tag"].values))
n_tags = len(tags)


# ------------OVER-SAMPLING-----------
need_resampling = ["GoodsOrigin", "HSCode", "Tolerance"]

grouping = df.groupby(df["sentence #"])
new_df = grouping.filter(
    lambda x: (x["tag"].apply(lambda tag: tag in need_resampling)).any()
)
new_df["sentence #"].groupby(new_df["sentence #"]).count()
for i in range(126):
    last_sentence_nr = df["sentence #"].max()
    new_sentences = set(new_df["sentence #"])
    nr_sentences_by_group = len(set(new_df["sentence #"]))
    di = {}
    for val, i in enumerate(new_sentences):
        di[i] = val + last_sentence_nr + 2
    new_df["sentence #"] = new_df["sentence #"].map(di)
    df = df.append(new_df).reset_index()
    del df["index"]
# doar pentru verificare
sorted = list(df["sentence #"].groupby(df["sentence #"]).count())
sorted.sort(reverse=True)

import random

groups = [df for _, df in df.groupby("sentence #")]
random.shuffle(groups)
df = pd.concat(groups).reset_index(drop=True)


class SentenceGetter(object):
    def __init__(self, data):
        self.n_sent = 0
        self.data = df
        self.empty = False
        agg_func = lambda s: [
            (w, t) for w, t in zip(s["words"].values.tolist(), s["tag"].values.tolist())
        ]
        self.grouped = self.data.groupby("sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped[self.n_sent]
            self.n_sent += 1
            return s
        except:
            return None


getter = SentenceGetter(df)
getter.get_next()
sentences = getter.sentences

max_len = 300
word2idx = {w: i for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}

import pickle

# Save the embeddings
with open("../../models/embeddings/words_embeddings", "wb") as fp:
    pickle.dump(word2idx, fp)

with open("../../models/embeddings/tags_embeddings", "wb") as fp:
    pickle.dump(tag2idx, fp)

from keras.preprocessing.sequence import pad_sequences

X = [[word2idx[w[0]] for w in s] for s in sentences]

import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# so we have the same length
X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=n_words - 1)

y = [[tag2idx[w[1]] for w in s] for s in sentences]
y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])
y = [to_categorical(i, num_classes=n_tags) for i in y]

from sklearn.model_selection import train_test_split

from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from tensorflow.keras import regularizers

regulaziers = regularizers.L1L2(l1=1e-5, l2=1e-5)

input = Input(shape=(max_len,))
model = Embedding(input_dim=n_words, output_dim=50, input_length=max_len)(
    input
)  # 50-dim embedding
model = Dropout(0.3)(model)
model = Bidirectional(
    LSTM(
        units=200,
        return_sequences=True,
        recurrent_dropout=0.1,
        kernel_regularizer=regulaziers,
    )
)(
    model
)  # variational biLSTM
model = Dropout(0.3)(model)
model = Bidirectional(
    LSTM(
        units=200,
        return_sequences=True,
        recurrent_dropout=0.1,
        kernel_regularizer=regulaziers,
    )
)(
    model
)  # variational biLSTM
out = TimeDistributed(Dense(n_tags, activation="softmax"))(
    model
)  # softmax output layer

model = Model(input, out)

from tensorflow.keras.metrics import AUC, Precision, Recall

model.compile(
    optimizer="rmsprop",
    loss="categorical_crossentropy",
    metrics=["accuracy", AUC(), Precision(), Recall()],
)

history = model.fit(
    X,
    np.array(y),
    batch_size=128,
    epochs=20,
    validation_split=0.2,
    verbose=1,
)

model.save("../../models/tensorflow/NER_model_updated_v3.h5")
hist = pd.DataFrame(history.history)
hist.to_csv('../../data/model_history.csv')