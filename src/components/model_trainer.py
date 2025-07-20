import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from src.utils import evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class Model_Trainer:
    def __init__(self):
        self.model_trainer = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('Splitting training and test input data')
            X_train, y_train, X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
                'Random Forest': RandomForestRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Linear Regression': LinearRegression(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'K-Neighbors Classifier': KNeighborsRegressor(),
                'CatBoosting Classifier': CatBoostRegressor(),
                'XGBClassifier': XGBRegressor(),
                'Adaboost Classifier' : AdaBoostRegressor(),
            }

            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)

            #To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            #To get best model name
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]
            print(best_model)
            if best_model_score <=0.6:
                raise CustomException("No best model found")
            logging.info("Best found model on both training and test data")

            save_object(file_path=self.model_trainer.trained_model_file_path,obj=best_model)
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test,predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e,sys)

