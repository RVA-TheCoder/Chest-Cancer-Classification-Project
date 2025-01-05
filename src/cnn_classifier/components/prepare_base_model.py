import os
from urllib import request
from zipfile import ZipFile
import tensorflow as tf
from cnn_classifier.entity.config_entity import PrepareBasseModelConfig
from pathlib import Path



class PrepareBaseModel:

    def __init__(self, config:PrepareBasseModelConfig):

        self.config=config

    @staticmethod
    def save_model(path:Path, model:tf.keras.Model):

        model.save(path)

    
    def get_base_model(self):

        self.model=tf.keras.applications.vgg16.VGG16(
            include_top=self.config.params_include_top,
            weights=self.config.params_weights,
            input_shape=self.config.params_image_size
        )

        # calling static method(save_model) to save the base model
        self.save_model(path=self.config.base_model_path,
                        model=self.model)
        
    
    @staticmethod
    def _prepare_full_model(model, classes, freeze_all:bool, freeze_till:int, learning_rate, input_shape):

        # Freeze all layers if freeze_all is True
        if freeze_all:
            model.trainable = False  # Correctly set layer.trainable
        
        # Freeze layers up to a specific number if freeze_till is provided
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False  # Correctly set layer.trainable

        # Creating new model on top
        inputs=tf.keras.Input(shape=input_shape, name="input_layer")

        # Apply Rescaling
        rescale_layer=tf.keras.layers.Rescaling(scale=1/255., name="rescaling_layer")
        x = rescale_layer(inputs)

        # VGG16 model
        x = model(x, training=False)

        # Hidden layers
        flatten_layer = tf.keras.layers.Flatten(name="flatten_layer")(x)

        # hidden_layer1 = tf.keras.layers.Dense(
        #                                     units=10,
        #                                     activation="sigmoid",
        #                                     name="hidden_layer1"
        #                                     )(flatten_layer)

        # hidden_layer2=tf.keras.layers.Dense(units=10, activation="relu", name="hidden_layer2")(hidden_layer1)

        #output_layer=tf.keras.layers.Dense(units=1, activation="sigmoid", name="output_layer")(hidden_layer2)

        hidden_layer1=tf.keras.layers.Dense(units=5, activation="relu", name="hidden_layer1")(flatten_layer)

        output_layer=tf.keras.layers.Dense(units=1, activation="sigmoid", name="output_layer")(hidden_layer1)

        full_model = tf.keras.models.Model(
                                  inputs=inputs,
                                  outputs=output_layer
                                  )
        
        full_model.summary()
        
        return full_model

        
    def custom_base_model(self):

        self.full_model=self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate,
            input_shape=self.config.params_image_size

            )
        
        self.save_model(path=self.config.custom_base_model_path,
                        model=self.full_model)




