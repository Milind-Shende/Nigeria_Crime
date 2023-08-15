import os
from NigeriaMLflow import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from NigeriaMLflow.entity.config_entity import DataTransformationConfig




class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    
    ## Note: You can add different data transformation techniques such as Scaler, PCA and all
    #You can perform all kinds of EDA in ML cycle here before passing this data to the model

    # I am only adding train_test_spliting cz this data is already cleaned up


    def train_test_spliting(self):
        data = pd.read_csv(self.config.data_path)

        logger.info("Pop the 'is_attack' column")
        # Pop the 'attack_type' column
        attack_type_column = data.pop('is_attack')

        logger.info("Append the 'is_attack' column back at the end")
        # Append the 'attack_type' column back at the end
        data['is_attack'] = attack_type_column

        logger.info(f"Checking unique value for is_attack{data['is_attack'].value_counts()}")
        data['is_attack'].value_counts()

        
        # data.rename(columns={'State':'state'},inplace=True)
        # logger.info(f"Checking data columns name {data.columns}")

        # Split the data into training and test sets. (0.80, 0.20) split.
        logger.info("Split the data into training and test sets. (0.80, 0.20) split")
        train, test = train_test_split(data,test_size=.20,random_state=42)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)

        logger.info("Splited data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)