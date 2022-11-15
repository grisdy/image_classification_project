import pandas as pd
import os
import logging as log

from common.config import get_config
from src.constant import Path

from tensorflow.keras.preprocessing.image import ImageDataGenerator


class Datagen:
    def __init__(self):
        self.config = get_config()
        self.dim = self.config['image']['self.dim']
        self.batch_size = self.config['image']['batch_size']
        
    def tr_val_gen(self, path: str = Path.TRAIN_PATH, aug = True):

        if aug:    
            log.info("Getting Data with augmentation")

            datagen_aug = ImageDataGenerator(
            rescale=1/255,
            width_shift_range = 0.2,
            height_shift_range = 0.2,
            zoom_range = 0.2,
            vertical_flip = True,
            validation_split = 0.2
             )

            train_generators = datagen_aug.flow_from_directory(path, 
                                            batch_size=batch_size,                                            
                                            target_size=self.dim,
                                            shuffle=True,
                                            seed=77,
                                            class_mode='categorical',
                                            subset='training'
                                           )
    
            validation_generators = datagen_aug.flow_from_directory(path, 
                                            batch_size=batch_size,
                                            target_size=self.dim,
                                            shuffle=False,
                                            class_mode='categorical',
                                            subset='validation'
                                           )
            return train_generators, validation_generators

        #Data Without Augmentation
        log.info("Getting Data w/o augmentation")

        datagen = ImageDataGenerator(
        rescale=1/255,
        validation_split = 0.2
            )
        train_generators = datagen.flow_from_directory(path, 
                                        batch_size=batch_size,                                            
                                        target_size=self.dim,
                                        shuffle=True,
                                        seed=77,
                                        class_mode='categorical',
                                        subset='training'
                                        )

        validation_generators = datagen.flow_from_directory(path, 
                                        batch_size=batch_size,
                                        target_size=self.dim,
                                        shuffle=False,
                                        class_mode='categorical',
                                        subset='validation'
                                        )
        return train_generators, validation_generators

    
    def test_gen(path, self.dim, batch_size):
        datagen = ImageDataGenerator(rescale=1/255)
        
        test_generators = datagen.flow_from_directory(path, 
                                                batch_size=batch_size,
                                                class_mode='categorical',
                                                target_size=self.dim,
                                                shuffle=False
                                                )
        return test_generators


