import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
import os,json
import pickle
import dill 
from src.utils import load_object
class PredictPipeline:
    def __init__(self):
        pass

    def predict_price(self,location,total_sqft,bhk): 
        try:

            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','proprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path) 
            try:
                loc_index = __data_columns.index(location.lower())
            except:
                loc_index = -5  
            x = np.zeros(157)
            #data_scaled=preprocessor.cat_pipleline 
            sqdt_scaled=preprocessor.named_transformers_['num_pipeline'].transform([[total_sqft,bhk]])
            #bhk_scaled = preprocessor.named_transformers_['num_pipeline'].transform([[bhk]])

            x[0:2]=sqdt_scaled.flatten()
            if loc_index>=0:
                x[loc_index] = 1

            return round(model.predict([x])[0])
            #return data_scaled.shape,data_scaled
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(self):
        pass

    def load_data(self):
        try:
            print("loading saved artifacts...start")
            global  __data_columns
            global __locations

            with open("templates\columns.json", "r") as f:
                __data_columns = json.load(f)['data_columns']
                __locations = __data_columns[3:]  # first 3 columns are sqft, bath, bhk

            global __model
            with open('artifacts/model.pkl', 'rb') as f:
                __model = pickle.load(f)
            print("loading saved artifacts...done")
        except Exception as e:
            raise CustomException(e, sys)
        
    def get_location_names(self):
        return __locations

    def get_data_columns(self):
        return __data_columns
    
    '''def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "total_sqft":[self.total_sqft],
                "bhk":[self.bhk],
                "location":[self.location.strip('"')]                
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)
    '''