from src.components.data_ingestion import DataIngestion
from src.components.data_ingestion import DataTransformation
from src.components.model_trainer import ModelTrainer

def main():
    ingestion = DataIngestion()
    df, img_size = ingestion.initiate_data_ingestion()

    train_df, val_df, test_df = ingestion.train_test_split(df)

    transformer = DataTransformation()
    train_ds, val_ds, test_ds = transformer.get_dataset_from_dataframe(
        train_df, val_df, test_df
    )

    trainer = ModelTrainer()
    trainer.build_and_train_model(img_size, train_ds, val_ds, test_ds)

if __name__ == "__main__":
    main()