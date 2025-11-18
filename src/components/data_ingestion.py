import os 
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

@dataclass 
class DataIngestionConfig:
    main_df_path: str = os.path.join("artifacts", "main_dataset.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        self.main_df = []
        self.IMG_SIZE = 200

    def initiate_data_ingestion(self):
        logging.info("Data ingestion is started.")

        try:
            data_transformation = DataTransformation()
            dataset_path = "notebooks/data/"

            self.main_df = data_transformation.create_utkface_dataframe(dataset_path)
            self.main_df["age_group_id"] = self.main_df["age_group"].cat.codes

            self.main_df["age_weight"] = self.main_df["age_group_id"].map(
                self.calculate_agegroup_class_weights())

            return self.main_df, self.IMG_SIZE
            
            


        except Exception as e:
            raise CustomException(e, sys)
        
    def calculate_agegroup_class_weights(self):
        classes = np.unique(self.main_df["age_group_id"])
        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=self.main_df["age_group_id"])

        age_weight = dict(zip(classes, weights))
        age_weight = {int(k): float(v) for k, v in age_weight.items()}

        return age_weight
    
    def train_test_split(self, df):
        logging.info("Data train, test is splitting started.")
        try: 
            train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True, stratify=df["age_group_id"])
            val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, shuffle=True, stratify=temp_df["age_group_id"])

            return(train_df, val_df, test_df)
        
        except Exception as e:
            raise CustomException(e,sys)
        
    
    


