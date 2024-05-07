from typing import Any
import tenso
from .custom import Custom
import thingsvision.utils.models.align.efficientnet as efficientnet


class Kakaobrain_Align(Custom):
    def __init__(self, device, parameters) -> None:
        super().__init__(device)
        self.backend = "tf"

    def create_model(self) -> Any:
        def create_vision_encoder(img_encoder_name, embed_dim):
            map = {
                'efficientnet-b0': efficientnet.EfficientNetB0,
                'efficientnet-b1': efficientnet.EfficientNetB1,
                'efficientnet-b2': efficientnet.EfficientNetB2,
                'efficientnet-b3': efficientnet.EfficientNetB3,
                'efficientnet-b4': efficientnet.EfficientNetB4,
                'efficientnet-b5': efficientnet.EfficientNetB5,
                'efficientnet-b6': efficientnet.EfficientNetB6,
                'efficientnet-b7': efficientnet.EfficientNetB7,
            }
            model = map[img_encoder_name.lower()](include_top=False, classifier_activation=None, pooling='avg',
                                                  weights=None)
            model.trainable = True

            inputs = tf.keras.layers.Input(shape=(289, 289, 3), name="image_input")
            embeddings = model(inputs)
            if img_encoder_name == 'efficientnet-b7':
                outputs = embeddings
            else:
                outputs = tf.keras.layers.Dense(embed_dim, dtype=tf.float32)(embeddings)

            return tf.keras.Model(inputs, outputs, name='vision_encoder')

        return model, None
