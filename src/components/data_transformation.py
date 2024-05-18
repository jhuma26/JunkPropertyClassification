import sys
import os
from dataclasses import dataclass
# current_dir = os.path.dirname(os.path.realpath('E:\Data science projects\project2\src\components\data_ingestion.py'))
# project_root = os.path.abspath(os.path.join(current_dir, "E:\Data science projects\project2"))
# sys.path.append(project_root)
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler,MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self,categorical_columns,float_columns,int_columns):
        '''
        This function is responsible for data trnasformation
        
        '''
        try:
            # target_column = 'Junk'
            # floatcols = ['PriceIndex1','PriceIndex2','PriceIndex3','PriceIndex4','PriceIndex5','PriceIndex6','PriceIndex7','PriceIndex8','PriceIndex9']
            # for i in floatcols:
            #     train_df[i] = train_df[i].astype(float)
            # categorical_columns = [col for col in train_df.columns if train_df[col].dtype == 'object' and col != target_column]
            # float_columns = [col for col in train_df.columns if train_df[col].dtype == 'float64' and col != target_column]
            # int_columns = [col for col in train_df.columns if train_df[col].dtype == 'int64' and col != target_column]
            
            # logging.info(f"Removal of correlated columns from preprocessing object starts")

            # #correlated_columns = list(self.correlated_columns(train_df,0.8))
            # correlated_columns = self.correlated_columns(train_df.drop(columns=[target_column]), threshold=0.8)
            # train_df.drop(columns=correlated_columns,axis=1,inplace=True)
            # logging.info(f"Removal of correlated columns preprocessing object ends")

            print("categorical columns",categorical_columns)
            print("float columns", float_columns)
            print("int columns", int_columns)

            float_pipeline= Pipeline(
                steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler",MinMaxScaler())
                ]
            )
            int_pipeline= Pipeline(
                steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler",StandardScaler()),
                #("selector", SelectKBest(score_func=mutual_info_classif, k='all'))
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder(min_frequency =  2000, 
                                                 handle_unknown = 'ignore',
                                                 drop = 'first'))
                #('replace_inf', SimpleImputer(strategy='constant', fill_value=-9999)),

                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Float columns: {float_columns}")
            logging.info(f"Int columns: {int_columns}")

            preprocessor=ColumnTransformer(
                transformers=[
                ("cat_pipelines",cat_pipeline,categorical_columns),
                ("float_pipeline",float_pipeline,float_columns),
                ("int pipeline",int_pipeline,int_columns)
                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)

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
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            train_df.drop(columns=['EnvRating', 'PRIMEUNIT','Zip','ListDate','BuildYear'],axis=1,inplace=True)
            test_df.drop(columns=['EnvRating', 'PRIMEUNIT','Zip','ListDate','BuildYear'],axis=1,inplace=True)

            train_df.replace('missing', np.nan, inplace=True)
            test_df.replace('missing',np.nan, inplace=True)

            target_column = 'Junk'
            floatcols = ['PriceIndex1','PriceIndex2','PriceIndex3','PriceIndex4','PriceIndex5','PriceIndex6','PriceIndex7','PriceIndex8','PriceIndex9']
            for i in floatcols:
                train_df[i] = train_df[i].astype(float)
            categorical_columns = [col for col in train_df.columns if train_df[col].dtype == 'object' and col != target_column]
            float_columns = [col for col in train_df.columns if train_df[col].dtype == 'float64' and col != target_column]
            int_columns = [col for col in train_df.columns if train_df[col].dtype == 'int64' and col != target_column]
            
            print("*************************************************")
            print("Categorical columns: ",categorical_columns)
            print("Float columns: ",float_columns)
            print("Int columns: ",int_columns)
            print("*************************************************")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj=self.get_data_transformer_object(categorical_columns,float_columns,int_columns)

            input_feature_train_df=train_df.drop(columns=[target_column],axis=1)
            target_feature_train_df=train_df[target_column]

            input_feature_test_df=test_df.drop(columns=[target_column],axis=1)
            target_feature_test_df=test_df[target_column]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            # return(
            #     input_feature_train_df,
            #     input_feature_test_df,
            #     self.data_transformation_config.preprocessor_obj_file_path
            # )
        except Exception as e:
            raise CustomException(e,sys)