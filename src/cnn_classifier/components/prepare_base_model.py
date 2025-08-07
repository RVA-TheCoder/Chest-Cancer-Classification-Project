import os
from urllib import request
from zipfile import ZipFile
import tensorflow as tf

from pathlib import Path
from typing import Optional

from cnn_classifier.entity.config_entity import PrepareBaseModelConfig



class PrepareBaseModel:

    """
    Class responsible for preparing the base CNN model using VGG16 
    architecture, modifying it with custom layers, and saving both
    the base and customized models to disk.
    """

    def __init__(self, config:PrepareBaseModelConfig):
        
        """
        Initializes the PrepareBaseModel class with configuration parameters.

        Parameters : 
            (a) config (PrepareBaseModelConfig): Configuration dataclass instance containing model preparation parameters.
                                                 
        """

        self.config=config


    @staticmethod
    def save_model(path:Path, model:tf.keras.Model):
        
        """
        Saves the given Keras model to the specified file path.

        Parameters : 
            (a) path (Path): The path where the model will be saved.
            (b) model (tf.keras.Model): The Keras model to save.
        """

        model.save(path)

    
    def get_base_model(self):
        
        """
        Loads the VGG16 model as the base architecture with optional top layers
        and pre-trained weights as specified in the config. Then saves this base model
        to the specified path.
        """
        
        """
        In VGG16, the "16" refers to the number of weight layers 
        (i.e., layers with learnable parameters) in the model.
        
        Breakdown of VGG16:
            VGG16 consists of:

             - 13 Convolutional layers & 3 Fully Connected (Dense) layers

            Total: 13 + 3 = 16 layers with weights
            
            Hence, the name: VGG16
        
        """
        
        self.model=tf.keras.applications.vgg16.VGG16(
            include_top=self.config.params_include_top,
            weights=self.config.params_weights,
            input_shape=self.config.params_image_size
        )

        # calling static method(save_model) to save the base model
        self.save_model(path=self.config.base_model_path,
                        model=self.model
                        )
        
    
    @staticmethod
    def _prepare_full_model(model,
                            classes,
                            freeze_all:bool,
                            freeze_till:Optional[int],
                            #learning_rate,   # option is given but not used
                            input_shape):

        """
        Modifies the base model by adding custom layers on top for classification.

        Parameters : 
            (a) model (tf.keras.Model): The base VGG16 model.
            (b) classes (int): Number of output classes for the classification task.
            (c) freeze_all (bool): If True, freeze all layers of the base model.
            (d) freeze_till (Optional[int]): If set, freeze layers up to this index.
            (e) input_shape (tuple): Shape of input image, e.g., (224, 224, 3).

        Returns:
            tf.keras.Model: A new Keras model with added layers on top of base model.
        """
        
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

        # Flatten layer
        flatten_layer = tf.keras.layers.Flatten(name="flatten_layer")(x)

        # Hidden layers
        # hidden_layer1 = tf.keras.layers.Dense(
        #                                     units=10,
        #                                     activation="sigmoid",
        #                                     name="hidden_layer1"
        #                                     )(flatten_layer)

        # hidden_layer2=tf.keras.layers.Dense(units=10, activation="relu", name="hidden_layer2")(hidden_layer1)

        #output_layer=tf.keras.layers.Dense(units=classes-1,
        #                                   activation="sigmoid",
        #                                   name="output_layer")(hidden_layer2)

        #hidden_layer1=tf.keras.layers.Dense(units=5, activation="relu", name="hidden_layer1")(flatten_layer)

        #output_layer=tf.keras.layers.Dense(units=classes-1, activation="sigmoid", name="output_layer")(hidden_layer1)


        output_layer=tf.keras.layers.Dense(units=classes,
                                           activation="softmax",
                                           name="output_layer")(flatten_layer)


        full_model = tf.keras.models.Model(
                                  inputs=inputs,
                                  outputs=output_layer
                                  )
        
        full_model.summary()
        
        return full_model

        
    def custom_base_model(self):
        
        """
        Creates a custom classification model using the base VGG16 model.
        Adds a flatten layer and softmax classification layer.
        Saves the customized model to disk at the specified path.
        """

        self.full_model=self._prepare_full_model(
                                        model=self.model,
                                        classes=self.config.params_classes,
                                        freeze_all=True,
                                        freeze_till=None,
                                        #learning_rate=self.config.params_learning_rate,
                                        input_shape=self.config.params_image_size

                                        )
        
        self.save_model(path=self.config.custom_base_model_path,
                        model=self.full_model)




