import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data trnasformation
        
        '''
        try:
            numerical_columns = ["total_sqft", "bhk"]
            categorical_columns = [
                "location",
            ]

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            train_df=train_df.dropna()
            test_df=test_df.dropna()
            train_df['bhk']=train_df['size'].astype(str).apply(lambda x: x.split(' ')[0])
            test_df['bhk']=test_df['size'].astype(str).apply(lambda x: x.split(' ')[0])
            train_df=train_df.drop(['size'],axis=1)
            test_df=test_df.drop(['size'],axis=1)

            def numbers(x):
                a = x.split('-')
                if len(a) == 2:
                    return (float(a[0]) + float(a[1])) / 2
                try:
                    return float(a[0])
                except:
                    return None
                
            train_df["total_sqft"] = train_df["total_sqft"].apply(numbers)
            test_df["total_sqft"] = test_df["total_sqft"].apply(numbers)
            train_df.dropna(inplace=True)
            test_df.dropna(inplace=True)
            train_df['bhk']=train_df['bhk'].apply(lambda x: int(x))
            test_df['bhk']=test_df['bhk'].apply(lambda x: int(x))

            train_df['location']=train_df['location'].apply(lambda x: x.strip())
            test_df['location']=test_df['location'].apply(lambda x: x.strip())

            a=train_df.groupby('location')['location'].agg('count').sort_values(ascending=False)
            others=a[a<10]
            train_df['location']=train_df['location'].apply(lambda x: "others" if x in others else x)

            b=test_df.groupby('location')['location'].agg('count').sort_values(ascending=False)
            others=b[b<10]
            test_df['location']=test_df['location'].apply(lambda x: "others" if x in others else x)

            train_df['price_sqft']=train_df['price']*100000/train_df['total_sqft']
            test_df['price_sqft']=test_df['price']*100000/test_df['total_sqft']

            train_df=train_df[train_df['price']<500]
            test_df=test_df[test_df['price']<500]

            train_df=train_df[train_df['total_sqft']<5100]
            test_df=test_df[test_df['total_sqft']<5100]
            
            train_df['bhk']

            train_df = train_df[train_df['bath']<train_df['bhk']+2]
            test_df=test_df[test_df['bath']<test_df['bhk']+2]
            

            def remove_pps_outliers(df):
                df_out = pd.DataFrame()
                for key, subdf in df.groupby('location'):
                    m = np.mean(subdf.price_sqft)
                    st = np.std(subdf.price_sqft)
                    reduced_df = subdf[(subdf.price_sqft>(m-st)) & (subdf.price_sqft<=(m+st))]
                    df_out = pd.concat([df_out,reduced_df],ignore_index=True)
                return df_out
            train_df = remove_pps_outliers(train_df)
            test_df = remove_pps_outliers(test_df)


            train_df.dropna(inplace=True)
            test_df.dropna(inplace=True)


            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="price"
            numerical_columns = ["total_sqft","bhk"]

            input_feature_train_df=train_df.drop(columns=[target_column_name,"bath","area_type","availability","society","balcony","price_sqft"],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name,"bath","area_type","availability","society","balcony","price_sqft"],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_df[target_column_name]=target_feature_train_df
            input_feature_test_df[target_column_name]=target_feature_test_df

            train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            test_arr=preprocessing_obj.transform(input_feature_test_df)


            #train_arr = np.c_[input_feature_train_arr,target_feature_train_df]
            #test_arr = np.c_[input_feature_test_arr,target_feature_test_df]
            

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
        except Exception as e:
            raise CustomException(e,sys)