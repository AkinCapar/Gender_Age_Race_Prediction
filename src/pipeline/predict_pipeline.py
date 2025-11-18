import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image

class Predictor:

    def __init__(self, model_path, img_size=200):
        self.model = load_model(model_path)
        self.img_size = img_size

        
        self.gender_map = {0: "Male", 1: "Female"}
        self.race_map = {
            0: "White",
            1: "Black",
            2: "Asian",
            3: "Indian",
            4: "Other"
        }
        self.age_map = {
            0: "<20",
            1: "20-30",
            2: "30-45",
            3: "45-60",
            4: "60+"
        }

    def preprocess_image(self, img):
        img = tf.image.resize(img, (200, 200))
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = np.expand_dims(img, axis=0)
        return img

    def predict(self, img):
        img = self.preprocess_image(img)

        preds = self.model.predict(img)

        gender_pred = np.argmax(preds["gender_output"], axis=1)[0]
        race_pred   = np.argmax(preds["race_output"], axis=1)[0]
        age_pred    = np.argmax(preds["age_output"], axis=1)[0]

        return {
            "gender": self.gender_map[gender_pred],
            "race": self.race_map[race_pred],
            "age_group": self.age_map[age_pred]
        }
    
    def letterbox(self, img):

        img = img.convert("RGB")  

        w, h = img.size
        desired = self.img_size

        
        scale = min(desired / w, desired / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        
        img = img.resize((new_w, new_h), Image.LANCZOS)

        
        new_img = Image.new("RGB", (desired, desired), (0, 0, 0))

        
        pad_x = (desired - new_w) // 2
        pad_y = (desired - new_h) // 2

        new_img.paste(img, (pad_x, pad_y))

        return new_img