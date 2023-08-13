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
from imblearn.over_sampling import SMOTE
import xgboost as xgb


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    
    def eval_metrics(self,actual, pred):
        f1=f1_score(actual,pred)
        recall = recall_score(actual, pred)
        precision = precision_score(actual, pred)
        accu_score=accuracy_score(actual,pred)  
        return f1,recall, precision,accu_score
    


    def log_into_mlflow(self):

        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)
        loaded_model_xgb = xgb.Booster()
        loaded_model_xgb.load_model(self.config.model_path)
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

        smt = SMOTE(random_state=42)
        X_train_fea, y_train_fea = smt.fit_resample(train_x, train_y)
        X_test_fea, y_test_fea = smt.fit_resample(test_x, test_y)


        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme


        with mlflow.start_run():

            Xdtrain = xgb.DMatrix(X_train_fea)
            predicted_qualities_train = loaded_model_xgb.predict(Xdtrain)
            predicted_labels_train = (predicted_qualities_train > 0.5).astype(int)


            (f1_train,recall_train, precision_train,accu_score_train) = self.eval_metrics(y_train_fea, predicted_labels_train)
            
            # Saving metrics as local
            scores_train = {"f1_train":f1_train,"recall_train": recall_train, "precision_train": precision_train, 'accu_score_train':accu_score_train}
            save_json(path=Path(self.config.metric_file_name_train), data=scores_train)


            Xdtest = xgb.DMatrix(X_test_fea)
            predicted_qualities_test = loaded_model_xgb.predict(Xdtest)
            predicted_labels_test = (predicted_qualities_test > 0.5).astype(int)
            (f1_test,recall_test, precision_test,accu_score_test) = self.eval_metrics(y_test_fea, predicted_labels_test)
            
            # Saving metrics as local
            scores_test = {"f1_test":f1_test,"recall_test": recall_test, "precision_test": precision_test,'accu_score_test':accu_score_test}
            save_json(path=Path(self.config.metric_file_name_test), data=scores_test)

            mlflow.log_params(self.config.all_params)

            mlflow.log_metric("f1_train", f1_train)
            mlflow.log_metric("recall_train", recall_train)
            mlflow.log_metric("precision_train", precision_train)
            mlflow.log_metric("accu_score_train", accu_score_train)
            


            mlflow.log_metric("f1_test", f1_test)
            mlflow.log_metric("recall_test", recall_test)
            mlflow.log_metric("precision_test", precision_test)
            mlflow.log_metric("accu_score_test", accu_score_test)
            
            


            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.xgboost.log_model(loaded_model_xgb, "model", registered_model_name="xgboostModel")
            else:
                mlflow.xgboost.log_model(loaded_model_xgb, "model")

    