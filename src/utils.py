import os
import sys

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

import dill

from src.exception import CustomException
from src.logger import logging

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj,file_obj)

    except:
        pass


def evaluate_models(x_train,y_train,x_test,y_test,models,params):
    try:
        report = {}
        for name,model in models.items():
            grid_search = GridSearchCV(estimator=model,param_grid=params[name],cv=5,n_jobs=-1,scoring='neg_mean_squared_error')
            grid_search.fit(x_train,y_train)
            
            best_model = grid_search.best_estimator_
            y_train_pred = best_model.predict(x_train)
            y_test_pred = best_model.predict(x_test)

            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[name+" "+str(grid_search.best_params_)] = test_model_score
        return report


    except Exception as e:
        raise CustomException(e,sys)