import sys 
import os
import glob
from dataclasses import dataclass

import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import tensorflow as tf



class DataTransformation:
    def __init__(self):
        pass
        

    def parse_utkface_filename(self, path):
    
        filename = os.path.basename(path)
        file_no_ext = os.path.splitext(filename)[0]

        parts = file_no_ext.split("_")

        if len(parts) != 4:
            return None, None, None  # skip

        try:
            age = int(parts[0])
            gender = int(parts[1])
            race = int(parts[2])
            return age, gender, race
        except:
            return None, None, None
        
        
    
    age_bins = [0, 20, 30, 45, 60, np.inf]
    age_labels = ['<20', '20-30', '30-45', '45-60', '60+']

    def create_utkface_dataframe(self, dataset_path):
        files = glob.glob(os.path.join(dataset_path, "*.jpg"))

        ages = []
        genders = []
        races = []
        file_paths = []

        for file in files:
            age, gender, race = self.parse_utkface_filename(file)

            if age is None:
                continue

            ages.append(age)
            genders.append(gender)
            races.append(race)
            file_paths.append(file)

        df = pd.DataFrame({
            'age': ages,
            'gender_id': genders,
            'race_id': races,
            'file': file_paths
        })

        df["age_group"] = pd.cut(df["age"], bins=self.age_bins, labels=self.age_labels)

        df["gender"] = df["gender_id"].map({0: "male", 1: "female"})
        df["race"] = df["race_id"].map({
            0: "white",
            1: "black",
            2: "asian",
            3: "indian",
            4: "other"
        })

        return df
    
    
    def parse_train(self, path, g, r, a, w):
        img = self.load_image(path)

        labels = {
            "gender_output": g,
            "race_output": r,
            "age_output": a
        }

        weights = {
            "gender_output": 1.0,
            "race_output": 1.0,
            "age_output": w
        }

        return img, labels, weights


    def parse_val(self, path, g, r, a):
        img = self.load_image(path)

        labels = {
            "gender_output": g,
            "race_output": r,
            "age_output": a
        }

        return img, labels
    
    def build_dataset(self, df, batch_size=32, training=False):
        paths = df["file"].values
        genders = df["gender_id"].values
        races = df["race_id"].values
        ages = df["age_group_id"].values

        if training:
            weights = df["age_weight"].values
            ds = tf.data.Dataset.from_tensor_slices((paths, genders, races, ages, weights))
            ds = ds.shuffle(len(df))
            ds = ds.map(lambda p, g, r, a, w: self.parse_train(p, g, r, a, w),
                        num_parallel_calls=tf.data.AUTOTUNE)
        else:
            ds = tf.data.Dataset.from_tensor_slices((paths, genders, races, ages))
            ds = ds.map(lambda p, g, r, a: self.parse_val(p, g, r, a),
                        num_parallel_calls=tf.data.AUTOTUNE)

        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds
    
    def get_dataset_from_dataframe(self, train_df, val_df, test_df):
        train_ds = self.build_dataset(train_df, training=True)
        val_ds   = self.build_dataset(val_df, training=False)
        test_ds  = self.build_dataset(test_df, training=False)

        return (train_ds, val_ds, test_ds)
    

    def load_image(self, path):
        IMG_SIZE = 200
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)

        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))

        return img