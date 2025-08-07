# Update the src/cnn_classifier/components.
import os
from urllib import request
from pathlib import Path
from zipfile import ZipFile
import time

import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory as Images

from cnn_classifier.entity.config_entity import TrainingConfig



class Training:
    
    """
    Handles the training process for a CNN model using TensorFlow/Keras.

    This includes:
        - Loading a pre-defined base model.
        - Preprocessing image data.
        - Applying optional data augmentation.
        - Training and saving the final trained model.
    """

    def __init__(self, config:TrainingConfig):
        
        """
        Initializes the Training class with a given configuration.

        Parameters : 
            (a)config (TrainingConfig): Configuration object containing all training parameters and paths.
        """
        
        self.config=config

        #print("Training Config : ", self.config)


    def get_custom_base_model(self):
        
        """
        Loads the base model architecture from the specified file path.
        This model acts as the backbone for further training or fine-tuning.
        """

        self.model=tf.keras.models.load_model( 
                    self.config.custom_base_model_path
                    )
    
    @staticmethod
    def save_model(path:Path, model:tf.keras.Model):
        
        """
        Saves the trained model to a specified path.

        Parameters : 
            (a) path (Path): Destination path to save the model.
            (b) model (tf.keras.Model): Keras model instance to be saved.
        """

        model.save(path)


    def preprocess_data(self):
        
        """
        Loads and preprocesses training and testing image datasets from directories.
        Converts them into batched TensorFlow datasets suitable for model training.
        """


        # Load images and split them into batches
        # color_mode='rgb' by default (i.e., number of input image channels =3 which is what required for VGG16 model)
        self.images_train = Images(
                            directory=self.config.training_data,
                            labels='inferred',
                            label_mode="categorical",
                            image_size = self.config.params_image_size[:-1],
                            batch_size = self.config.params_batch_size
                            )

        # color_mode='rgb' by default (i.e., number of input image channels =3 which is what required for VGG16 model)
        self.images_test = Images(
                            directory=self.config.testing_data,
                            labels='inferred',
                            label_mode="categorical",
                            image_size = self.config.params_image_size[:-1],
                            batch_size = self.config.params_batch_size
                            )

    def train(self):
        
        """
        Trains the CNN model using the preprocessed data and the specified configuration.

        If data augmentation is enabled, it applies random transformations
        such as flip, translation, rotation, and zoom to the training input pipeline.

        After training, the model is saved to the specified path.
        """

        if self.config.params_is_augmentation:

            data_aug_layers = tf.keras.Sequential(
                [
                    tf.keras.layers.RandomFlip(mode='horizontal_and_vertical' ,
                                               
                                               name="Random_flip_layer" ),
                    tf.keras.layers.RandomTranslation(height_factor=(-0.1,.1),
                                                    width_factor=(-0.1,0.1),
                                                    fill_mode="reflect", 
                                                    name= "Random_translation_layers"),
                    tf.keras.layers.RandomRotation(factor=(-0.1,0.1), 
                                                   name="Random_rotation_layer"),
                    tf.keras.layers.RandomZoom(height_factor=(0.1,0.1),
                                               width_factor=(0.1,0.1), 
                                               name="Random_zoom_layer" )
                ],
                name="data_augmentation_layers"
                    )
            
            # Creating new model on top
            inputs=tf.keras.Input(shape=self.config.params_image_size, name="input_layer")

            # Apply random data augmentation
            x = data_aug_layers(inputs )

            Outputs = self.model(x, training=False)

            self.full_model = tf.keras.models.Model(
                                  inputs=inputs,
                                  outputs=Outputs
                                  )
            
        else :
            self.full_model = self.model

        # Compiling the model
        self.full_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.params_learning_rate),
                                loss=tf.keras.losses.CategoricalCrossentropy(),
                                metrics=self.config.params_metrices
                                )   
        
        #Fit the model
        self.history_fm = self.full_model.fit(x=self.images_train,
                                              validation_data=self.images_test,
                                              epochs=self.config.params_epochs)


        self.save_model(
                        path=self.config.trained_model_path,
                        model=self.full_model
                       )






