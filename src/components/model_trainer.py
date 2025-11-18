import os 
import sys
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, Model
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Activation,
    Dropout,
    Flatten,
    Dense
)
from sklearn.metrics import classification_report

from src.exception import CustomException
from src.logger import logging


@dataclass

class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.h5")
    pass

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def build_and_train_model(self, IMG_SIZE, train_ds, val_ds, test_ds):
        model = self.build_model(IMG_SIZE)

        model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss={
            "gender_output": "sparse_categorical_crossentropy",
            "race_output": "sparse_categorical_crossentropy",
            "age_output": "sparse_categorical_crossentropy",
        },
        metrics={
            "gender_output": [
                "accuracy"
            ],
            "race_output": [
                "accuracy"
            ],
            "age_output": [
                "accuracy"
            ]
        })

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=25
        )

        model.save("artifacts/model.keras")

        self.evaluate_model(model, test_ds)

    def build_model(self, IMG_SIZE):
        inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

        x = Conv2D(16, (3, 3), padding="same")(inputs)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(32, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(64, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(128, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.3)(x)

        x = Flatten()(x)
        x = Dense(256, activation="relu")(x)
        x = Dropout(0.3)(x)

        gender = layers.Dense(128, activation="relu")(x)
        gender = layers.Dropout(0.2)(gender)
        gender_output = layers.Dense(2, activation="softmax", name="gender_output")(gender)

        race = layers.Dense(128, activation="relu")(x)
        race = layers.Dropout(0.2)(race)
        race_output = layers.Dense(5, activation="softmax", name="race_output")(race)

        age = layers.Dense(128, activation="relu")(x)
        age = layers.Dropout(0.2)(age)
        age_output = layers.Dense(5, activation="softmax", name="age_output")(age)


        model = Model(
        inputs=inputs,
        outputs={
            "gender_output": gender_output,
            "race_output": race_output,
            "age_output": age_output
            })
        
        return model
    
    def evaluate_model(self, model, test_ds):
        y_true_gender = []
        y_true_race = []
        y_true_age = []

        y_pred_gender = []
        y_pred_race = []
        y_pred_age = []

        for images, labels in test_ds:
            preds = model.predict(images, verbose=0)

            pred_gender = np.argmax(preds["gender_output"], axis=1)
            pred_race   = np.argmax(preds["race_output"], axis=1)
            pred_age    = np.argmax(preds["age_output"], axis=1)

            y_pred_gender.extend(pred_gender)
            y_pred_race.extend(pred_race)
            y_pred_age.extend(pred_age)

            y_true_gender.extend(labels["gender_output"].numpy())
            y_true_race.extend(labels["race_output"].numpy())
            y_true_age.extend(labels["age_output"].numpy())

        print("===== Gender Classification =====")
        print(classification_report(y_true_gender, y_pred_gender, target_names=["Male", "Female"]))

        print("\n===== Race Classification =====")
        race_names = ["White", "Black", "Asian", "Indian", "Other"]
        print(classification_report(y_true_race, y_pred_race, target_names=race_names))

        print("\n===== Age Group Classification =====")
        age_names = ["<20", "20-30", "30-45", "45-60", "60+"]
        print(classification_report(y_true_age, y_pred_age, target_names=age_names))