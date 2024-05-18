import sys
import os
from dataclasses import dataclass
# current_dir = os.path.dirname(os.path.realpath('E:\Data science projects\project2\src\components\data_ingestion.py'))
# project_root = os.path.abspath(os.path.join(current_dir, "E:\Data science projects\project2"))
# sys.path.append(project_root)
import numpy as np 
import pandas as pd

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class FeatureEngineeringConfig:
    featengineering_obj_file_path=os.path.join('artifacts',"feature_engineer.pkl")

class FeatureEngineering:
    def __init__(self):
        self.featureengineering_config = FeatureEngineeringConfig()

    def correlated_columns(self,dataset, threshold):
        logging.info(
                    f"correlated_columns() method called"
                )
        col_corr = set()  
        corr_matrix = dataset.corr()
        try:
            for i in range(len(corr_matrix.columns)):
                for j in range(i):
                    if (corr_matrix.iloc[i, j]) > threshold: 
                        colname = corr_matrix.columns[i]  
                        col_corr.add(colname)

            return col_corr
        except Exception as e:
            raise CustomException(e,sys)
        
    def feature_engineered(self,df_train,df_test):
        try:
            correlated_col = self.correlated_columns(df_train,0.8)
            print("correlated_col", correlated_col)
            input_corr_train_df=df_train.drop(columns=[correlated_col],axis=1)
            input_corr_test_df=df_test.drop(columns=[correlated_col],axis=1)
            #correlated_obj=self.get_data_transformer_object(df_train,0.8)
            target_column_name = 'Junk'
            logging.info(
                    f"Applying correlated object on training dataframe and testing dataframe."
                )
            input_corr_train_arr=correlated_obj.fit_transform(input_corr_train_df)
            input_corr_test_arr=correlated_obj.transform(input_corr_test_df)

            target_corr_train_df=df_train[target_column_name]
            target_corr_test_df=df_test[target_column_name]

            train_arr = np.c_[
                    input_corr_train_arr, np.array(target_corr_train_df)
                ]
            test_arr = np.c_[input_corr_test_arr, np.array(target_corr_test_df)]

            logging.info(f"Saved feature engineered object.")

            save_object(

                    file_path=self.featureengineering_config.featengineeing_obj_file_path,
                    obj=correlated_obj

                )
            return (
                train_arr,
                test_arr,
                self.featureengineering_config.featengineering_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
