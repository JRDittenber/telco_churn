import os 
import sys 
import pandas as pd
from sklearn.model_selection import train_test_split

from telco_churn.entity.config_entity import DataIngestionConfig
from telco_churn.entity.artifact_entity import DataIngestionArtifact

from telco_churn.exceptions import custom_exception
from telco_churn.logger import logging

from telco_churn.database_access.db_extract import TelcoData


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        """
        :param data_ingestion_config: Configuration for data ingestion.
        
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise custom_exception(e, sys)


    def export_data_into_feature_store(self) -> pd.DataFrame:
        """
        Description: This method exports data from MongoDB to a csv file.

        output: data is returned as artifact of a data ingestion component
        on failure: write an exception log and then raise a custom exception
        """
        try: 
            logging.info("Exporting data from MongoDB.")
            telco_data = TelcoData()
            dataframe = telco_data.export_collection_to_dataframe(collection_name=self.data_ingestion_config.collection_name)
            logging.info(f"Shape of dataframe: {dataframe.shape}")
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"Saving exported data into feature store at {feature_store_file_path}.")
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe
        except Exception as e:
            raise custom_exception(e, sys)
        

    def split_data_as_train_test(self, dataframe: pd.DataFrame) -> None:
        """
        Description: split the dataframe into train and test based on split ratio

        output: Folder created in s3 bucket with train and test data
        on failure: write an exception log and then raise a custom exception
        """
        logging.info("Entering the split_data_as_train_test method of Data_Ingestion class.")

        try:
            if dataframe.empty:
                raise ValueError("The DataFrame is empty. Please check the data loading process.")

            train_set, test_set = train_test_split(dataframe, test_size=self.data_ingestion_config.train_test_split_ratio)
            logging.info(f'Performed the train test split with ratio: {self.data_ingestion_config.train_test_split_ratio}')
            logging.info(f"Train set shape: {train_set.shape}")
            logging.info(f"Test set shape: {test_set.shape}")
            logging.ingo("Exiting the split_data_as_train_test method of Data_Ingestion class.")

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            logging.info(f"Exporting train and test data to {self.data_ingestion_config.training_file_path} and {self.data_ingestion_config.testing_file_path}.")
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)
            
            logging.info(f"Train and test data exported successfully to {self.data_ingestion_config.training_file_path} and {self.data_ingestion_config.testing_file_path}.")

        except Exception as e:
            raise custom_exception(e, sys) from e
        

        def initiate_data_ingestion(self) -> DataIngestionArtifact:
            """
            Description: This method initiates the data ingestion process.

            output: train set and test set are returned as artifact of a data ingestion component
            on failure: write an exception log and then raise a custom exception
            """
            logging.info("Entering the initiate_data_ingestion method of Data_Ingestion class.")
           
           try:
            dataframe = self.export_data_into_feature_store()
            logging.info("Got the data from MonogDB")

            if dataframe.empty:
                raise ValueError("The DataFrame is empty. Please check the data loading process.")
            
            self.split_data_as_train_test(dataframe)
            logging.info("Split the data into train and test sets.")

            logging.info("Exiting the initiate_data_ingestion method of Data_Ingestion class.")

            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                testing_file_path=self.data_ingestion_config.testing_file_path,
            )
            logging.info(f"Data ingestion artifact created: {data_ingestion_artifact}.")
            return data_ingestion_artifact
        except Exception as e:
            raise custom_exception(e, sys) from e
            