from dataclasses import dataclass
import numpy as np
import os,sys
from src.logger import logging
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.utils import save_object,evaluate_models
from src.exception import CustomException


@dataclass 
class ModelTrainerconfig():
    trained_model_file_path=os.path.join("artifacts","model.pkl")



class Modeltrainer():
    def __init__(self):
        self.model_trainer_config=ModelTrainerconfig()

    def Initiatemodeltrianer(self,x_train,x_test,y_train,y_test):
        try:
            logging.info("Split training and test input data")
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            print("model_trainer")




            model_report:dict=evaluate_models(X_train=x_train,y_train=y_train,X_test=x_test,y_test=y_test,models=models)
            
                ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

                ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.75:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(x_test)
                
            r2_square = r2_score(y_test, predicted)
            return r2_square
            
        except Exception as e:
             raise CustomException(e,sys)