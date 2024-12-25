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
    def _prepare_full_model(model, classes, freeze_all:bool, freeze_till:int, learning_rate):

        # Freeze all layers if freeze_all is True
        if freeze_all:
            model.trainable = False  # Correctly set layer.trainable
        
        # Freeze layers up to a specific number if freeze_till is provided
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False  # Correctly set layer.trainable

        custom_model=tf.keras.models.Sequential(
            model.layers,
            [tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=10, activation="relu", name="Hidden_layer1"),
            tf.keras.layers.Dense(units=5,activation="relu", name="Hiddem_layer2"),
            tf.keras.layers.Dense(units=classes, activation="sigmoid", name="Output_layer")

            ])
        
        # Compiling the model
        custom_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                             loss="binary_crossentropy",
                             metrics=["accuracy"])
        
        custom_model.summary()
        
        return custom_model

        
    def custom_base_model(self):

        self.full_model=self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate

            )
        
        self.save_model(path=self.config.custom_base_model_path,
                        model=self.full_model)




