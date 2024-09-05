import os
import sys
from dataclasses import dataclass


from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainingCofig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingCofig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split train and test input data")
            x_train,y_train,x_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                'RandomForest': RandomForestRegressor(),
                'DecisionTree': DecisionTreeRegressor(),
                'GradientBoosting': GradientBoostingRegressor(),
                'LinearRegression': LinearRegression(),
                'KNeighbors': KNeighborsRegressor(),
                'XGBoost': XGBRegressor(),
                'AdaBoost': AdaBoostRegressor()
            }

            params = {
                'RandomForest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                },
                'DecisionTree': {
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                },
                'GradientBoosting': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                },
                'LinearRegression': {
                    'fit_intercept': [True, False]
                },
                'KNeighbors': {
                    'n_neighbors': [3, 5, 7, 9],
                },
                'XGBoost': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                },
                'CatBoost': {
                    'iterations': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'depth': [3, 5, 7]
                },
                'AdaBoost': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
            }

            model_report:dict = evaluate_models(x_train,y_train,x_test,y_test,models,params)

            logging.info('model Report:')
            logging.info(model_report)

            best_model_score = max(sorted(model_report.values()))\
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name.split()[0]]
            if(best_model_score < 0.6):
                raise CustomException("No best model found")
            logging.info("best found model on both training and testing dataset")

            best_model = best_model.fit(x_train,y_train)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info('object was saved')
            predicted = best_model.predict(x_test)
            r2_square = r2_score(y_test,predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e,sys)
         
        