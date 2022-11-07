import tensorflow as tf
import keras

model = keras.models.load_model("../models/tensorflow/NER_model_updated_v2.h5")

tf.saved_model.save(model, "../models/tensorflow/tf_pb_model_v2")

import os

os.system(
    'python -m tf2onnx.convert --saved-model ../models/tensorflow/tf_pb_model_v2 --output "../models/onnx/model_v2.onnx"'
)
