import os
import pandas as pd
from sklearn.metrics import  f1_score,accuracy_score,recall_score,precision_score,confusion_matrix,roc_curve,auc
from urllib.parse import urlparse
import mlflow
import mlflow.xgboost
import numpy as np
import joblib
from NigeriaMLflow.entity.config_entity import ModelEvaluationConfig
from NigeriaMLflow.utils.common import save_json
from pathlib import Path
from NigeriaMLflow import logger


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    
    def eval_metrics(self,actual, pred):
        f1=f1_score(actual,pred,average='macro')
        recall = recall_score(actual, pred,average='macro')
        precision = precision_score(actual, pred,average='macro')  
        return f1,recall, precision
    


    def log_into_mlflow(self):

        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)
        transformer = joblib.load(self.config.transformer_path)
        target = joblib.load(self.config.target_path)

        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[self.config.target_column]
        test_y = test_data[self.config.target_column]

    
        logger.info("Transforming the Train and Test")
        train_x = transformer.fit_transform(train_x)
        test_x = transformer.transform(test_x)

        logger.info("Scaling train_y and test_y")
        
        train_y=target.fit_transform(train_y)
        test_y=target.fit_transform(test_y)


        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme


        with mlflow.start_run():

            predicted_qualities_train = model.predict(train_x)

            (f1_train,recall_train, precision_train) = self.eval_metrics(train_y, predicted_qualities_train)
            
            # Saving metrics as local
            scores_train = {"f1_train":f1_train,"recall_train": recall_train, "precision_train": precision_train}
            save_json(path=Path(self.config.metric_file_name_train), data=scores_train)



            predicted_qualities_test = model.predict(test_x)

            (f1_test,recall_test, precision_test) = self.eval_metrics(test_y, predicted_qualities_test)
            
            # Saving metrics as local
            scores_test = {"f1_test":f1_test,"recall_test": recall_test, "precision_test": precision_test}
            save_json(path=Path(self.config.metric_file_name_test), data=scores_test)

            mlflow.log_params(self.config.all_params)

            mlflow.log_metric("f1_train", f1_train)
            mlflow.log_metric("recall_train", recall_train)
            mlflow.log_metric("precision_train", precision_train)
            


            mlflow.log_metric("f1_test", f1_test)
            mlflow.log_metric("recall_test", recall_test)
            mlflow.log_metric("precision_test", precision_test)
            
            


            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.xgboost.log_model(model, "model", registered_model_name="xgboostModel")
            else:
                mlflow.xgboost.log_model(model, "model")

    