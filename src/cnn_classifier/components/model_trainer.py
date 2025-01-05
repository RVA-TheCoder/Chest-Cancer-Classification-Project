# Update the src/cnn_classifier/components.
import os
from urllib import request
from zipfile import ZipFile
import tensorflow as tf
import time
from tensorflow.keras.utils import image_dataset_from_directory as Images
from pathlib import Path
from cnn_classifier.entity.config_entity import TrainingConfig



class Training:

    def __init__(self, config:TrainingConfig):
        self.config=config

        #print("Training Config : ", self.config)

    def get_custom_base_model(self):

        self.model=tf.keras.models.load_model( 
                    self.config.custom_base_model_path
                    )
    
    @staticmethod
    def save_model(path:Path, model:tf.keras.Model):

        model.save(path)

    def preprocess_data(self):

        # Load images and split them into batches
        self.images_train = Images(
                            directory=self.config.training_data,
                            labels='inferred',
                            image_size = self.config.params_image_size[:-1],
                            batch_size = self.config.params_batch_size
                            )

        self.images_test = Images(
                            directory=self.config.testing_data,
                            labels='inferred',
                            image_size = self.config.params_image_size[:-1],
                            batch_size = self.config.params_batch_size
                            )
        
    def train(self):

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
                           loss="binary_crossentropy",
                           metrics=["accuracy"])
        
        #Fit the model
        self.history_fm = self.full_model.fit(x=self.images_train,
                                              validation_data=self.images_test,
                                              epochs=self.config.params_epochs)


        self.save_model(
                        path=self.config.trained_model_path,
                        model=self.full_model
                       )






