import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        
        '''
        try:
            numerical_columns = ["total_sqft", "bhk"]
            categorical_columns = ["location"]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipelines", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self,raw_data_path):

        try:
            data=pd.read_csv(raw_data_path)
            
            # Combine train and test datasets for preprocessing
            #combined_df = pd.concat([train_df, test_df], ignore_index=True)
            combined_df=data.copy()
            combined_df = combined_df.dropna()
            combined_df['bhk'] = combined_df['size'].astype(str).apply(lambda x: x.split(' ')[0])
            combined_df = combined_df.drop(['size'], axis=1)

            def numbers(x):
                a = x.split('-')
                if len(a) == 2:
                    return (float(a[0]) + float(a[1])) / 2
                try:
                    return float(a[0])
                except:
                    return None

            combined_df["total_sqft"] = combined_df["total_sqft"].apply(numbers)
            combined_df.dropna(inplace=True)
            combined_df['bhk'] = combined_df['bhk'].apply(lambda x: int(x))
            combined_df['location'] = combined_df['location'].apply(lambda x: x.strip())

            a = combined_df.groupby('location')['location'].agg('count').sort_values(ascending=False)
            others = a[a < 10]
            combined_df['location'] = combined_df['location'].apply(lambda x: "others" if x in others else x)

            combined_df['price_sqft'] = combined_df['price']*100000/combined_df['total_sqft']

            combined_df = combined_df[combined_df['price'] < 500]
            combined_df = combined_df[combined_df['total_sqft'] < 5100]
            combined_df = combined_df[combined_df['bath'] < combined_df['bhk'] + 2]

            def remove_pps_outliers(df):
                df_out = pd.DataFrame()
                for key, subdf in df.groupby('location'):
                    m = np.mean(subdf.price_sqft)
                    st = np.std(subdf.price_sqft)
                    reduced_df = subdf[(subdf.price_sqft>(m-st)) & (subdf.price_sqft<=(m+st))]
                    df_out = pd.concat([df_out,reduced_df],ignore_index=True)
                return df_out
            cleaned_df=remove_pps_outliers(combined_df)

            target_column_name="price"
            #train_data,test_data=train_test_split(cleaned_df,test_size=0.2,random_state=42)
            x=cleaned_df.drop(columns=[target_column_name,"bath","area_type","availability","society","balcony","price_sqft"],axis=1)
            y=cleaned_df[target_column_name]
            x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            #target_column_name="price"

            #input_feature_train_df=train_data.drop(columns=[target_column_name,"bath","area_type","availability","society","balcony","price_sqft"],axis=1)
            #y_train=train_data[target_column_name]

            #input_feature_test_df=test_data.drop(columns=[target_column_name,"bath","area_type","availability","society","balcony","price_sqft"],axis=1)
            #y_test=test_data[target_column_name]

            logging.info("Applying preprocessing object on the dataset.")

        

            x_train=preprocessing_obj.fit_transform(x_train)
            x_test=preprocessing_obj.transform(x_test)
            


            print(x_train.shape)
            print(x_test.shape)
            print(y_train.shape)
            print(y_test.shape)

            logging.info("Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                x_train,
                x_test,
                y_train,
                y_test,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        
        except Exception as e:
            raise CustomException(e, sys)
