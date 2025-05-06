import os
import sys

import numpy as np
import pandas as pd
from telco_churn.entity.config_entity import TelcoPredictorConfig
from telco_churn.entity.s3_estimator import TelcoEstimator
from telco_churn.exceptions import custom_exception
from telco_churn.logger import logging
from telco_churn.utils.main_utils import read_yaml_file
from pandas import DataFrame




class TelcoData:
    def __init__(self,
                SeniorCitizen,
                Dependents,
                tenure,
                MultipleLines,
                InternetService,
                OnlineSecurity,
                TechSupport,
                StreamingTV,
                StreamingMovies,
                Contract,
                PaperlessBilling,
                PaymentMethod,
                MonthlyCharges, 
                TotalCharges
                ):
        """
        Telco Data constructor
        Input: all features of the trained model for prediction
        """
        try:
            self.SeniorCitizen = SeniorCitizen
            self.Dependents = Dependents
            self.tenure = tenure
            self.MultipleLines = MultipleLines
            self.InternetService = InternetService
            self.OnlineSecurity = OnlineSecurity
            self.TechSupport = TechSupport
            self.StreamingTV = StreamingTV
            self.StreamingMovies = StreamingMovies
            self.Contract = Contract
            self.PaperlessBilling = PaperlessBilling
            self.PaymentMethod = PaymentMethod
            self.MonthlyCharges = MonthlyCharges
            self.TotalCharges = TotalCharges



        except Exception as e:
            raise custom_exception(e, sys) from e

    def get_telco_input_data_frame(self)-> DataFrame:
        """
        This function returns a DataFrame from TelcoData class input
        """
        try:
            
            telco_input_dict = self.get_usvisa_data_as_dict()
            return DataFrame(telco_input_dict)
        
        except Exception as e:
            raise custom_exception(e, sys) from e


    def get_telco_data_as_dict(self):
        """
        This function returns a dictionary from USvisaData class input 
        """
        logging.info("Entered get_usvisa_data_as_dict method as USvisaData class")

        try:
            input_data = {
                "SeniorCitizen": [self.SeniorCitizen],
                "Dependents": [self.Dependents],
                "tenure": [self.tenure],
                "MultipleLines": [self.MultipleLines],
                "InternetService": [self.InternetService],
                "OnlineSecurity": [self.OnlineSecurity],
                "TechSupport": [self.TechSupport],
                "StreamingTV": [self.StreamingTV],
                "StreamingMovies": [self.StreamingMovies],
                "Contract": [self.Contract],
                "PaperlessBilling": [self.PaperlessBilling],
                "PaymentMethod": [self.PaymentMethod],
                "MonthlyCharges": [self.MonthlyCharges],
                "TotalCharges": [self.TotalCharges],
            }

            logging.info("Created telco data dict")

            logging.info("Exited get_telco_data_as_dict method as TelcoData class")

            return input_data

        except Exception as e:
            raise custom_exception(e, sys) from e



class TelcoClassifier:
    def __init__(self,prediction_pipeline_config: TelcoPredictorConfig = TelcoPredictorConfig(),) -> None:
        """
        :param prediction_pipeline_config: Configuration for prediction the value
        """
        try:
            # self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise custom_exception(e, sys)

    def predict(self, dataframe) -> str:
        """
        This is the method of USvisaClassifier
        Returns: Prediction in string format
        """
        try:
            logging.info("Entered predict method of USvisaClassifier class")
            model = TelcoEstimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            result =  model.predict(dataframe)
            
            return result
        
        except Exception as e:
            raise custom_exception(e, sys)