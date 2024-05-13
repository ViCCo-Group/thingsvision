from typing import Any
import os
import tensorflow as tf
from .custom import Custom
from thingsvision.utils.checkpointing import download_file, get_keras_home
import thingsvision.utils.models.align.efficientnet as efficientnet
from tensorflow.keras.utils import get_file

DOWNLOAD_FILES = {
    'model-weights.index': 'https://huggingface.co/kakaobrain/coyo-align-b7-base/resolve/main/model-weights.index?download=true',
    'model-weights.data-00000-of-00001': 'https://huggingface.co/kakaobrain/coyo-align-b7-base/resolve/main/model-weights.data-00000-of-00001?download=true'
}


class ALIGN(tf.keras.models.Model):
    def __init__(self, image_encoder):
        super(ALIGN, self).__init__()
        self.image_encoder = image_encoder

    def call(self, inputs, training):
        image_features = self.image_encoder(inputs, training)
        return image_features


class Kakaobrain_Align(Custom):
    def __init__(self, device, parameters) -> None:
        super().__init__(device)
        self.backend = "tf"

    def _download_weights(self):
        model_folder = os.path.join(get_keras_home(), 'models', 'kakaobrain_align')
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
            for file_name, url in DOWNLOAD_FILES.items():
                get_file(origin=url, fname=os.path.join(model_folder, file_name))
        return model_folder

    def create_model(self) -> Any:
        model = tf.keras.applications.EfficientNetB7(include_top=False,
                                                     classifier_activation=None,
                                                     pooling='avg',
                                                     weights=None)
        model._name = 'image_encoder'
        model.trainable = False

        weights_path = self._download_weights()
        inputs = tf.keras.layers.Input(shape=(289, 289, 3), name="image_input")
        outputs = model(inputs)
        model = ALIGN(model)
        model.build(input_shape=(289, 289, 3))
        checkpoint = tf.train.Checkpoint(model)
        path = os.path.join(weights_path, 'model-weights')
        status = checkpoint.restore(path)
        print(status)
        #model.load_weights(os.path.join(weights_path, 'model-weights'), by_name=True, skip_mismatch=True)
        # print(model.layers)
        return model, tf.keras.applications.efficientnet.preprocess_input
