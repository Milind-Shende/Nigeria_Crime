import pandas as pd
import os
from NigeriaMLflow import logger
import joblib
from NigeriaMLflow.entity.config_entity import ModelTrainerConfig
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler,MinMaxScaler,StandardScaler,LabelEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.multioutput import ClassifierChain




class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[self.config.target_column]
        test_y = test_data[self.config.target_column]

        numeric_features=['multilingual_literacy', 'literacy', 'university_admission',
       'sanitation', 'electricity', 'state_unemployment', 'avg_household_size',
       'region_population_size', 'drinking_water', 'month', 'day',
       'isweekday', 'is_holiday']

        categorical_features =['State']

        # Numerical and Categorical Pipeline Transformation
        logger.info("Numerical and Categorical Pipeline Transformation")
        numeric_transformer = Pipeline(steps=[('Scaler', MinMaxScaler())])
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(sparse=True, handle_unknown='ignore'))])

        # Numerical and Categorical Column Transformation
        logger.info("Numerical and Categorical Column Transformation")
        transformer = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),
                                                      ('cat', categorical_transformer, categorical_features)])
        logger.info("Transforming the Train and Test")
        train_x = transformer.fit_transform(train_x)
        test_x = transformer.transform(test_x)

        logger.info("Scaling train_y and test_y")
        target_scaler=LabelEncoder()
        train_y=target_scaler.fit_transform(train_y)
        test_y=target_scaler.fit_transform(test_y)

        
        

       

        logger.info("define Xgboost Classification")
        xgb_model = XGBClassifier(n_estimators=self.config.n_estimators,
                                  scale_pos_weight=self.config.scale_pos_weight)

        logger.info("fit train_x and train_y")
        xgb_model.fit(train_x, train_y)

        logger.info("Creating a model pickle file")
        # joblib.dump(xgb_model, os.path.join(self.config.root_dir, self.config.model_name))
        xgb_model.save_model(os.path.join(self.config.root_dir, self.config.model_name))
        logger.info("Creating a transformer pickle file")
        joblib.dump(transformer, os.path.join(self.config.root_dir, self.config.transformer_name))
        logger.info("Creating a target pickle file")
        joblib.dump(target_scaler, os.path.join(self.config.root_dir, self.config.target_name))