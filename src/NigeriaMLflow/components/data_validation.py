import os
from NigeriaMLflow import logger
from NigeriaMLflow.entity.config_entity import DataValidationConfig
import pandas as pd
from scipy.stats import ks_2samp


class DataValiadtion:
    def __init__(self, config: DataValidationConfig):
        self.config = config


    def validate_all_columns(self) -> bool:
        try:
            validation_status = None
            logger.info("Reading of data from unzip_data_dir")
            data = pd.read_csv(self.config.unzip_data_dir)
            all_cols = list(data.columns)
            all_schema = self.config.all_schema

            # Check if all columns are present in the expected schema
            logger.info("Check if all columns are present in the expected schema")
            missing_cols = set(all_schema.keys()) - set(all_cols)
            if missing_cols:
                logger.warning(f"Missing columns in data: {missing_cols}")
                validation_status = False
                with open(self.config.STATUS_FILE, 'w') as f:
                    f.write(f"Validation status: {validation_status}")
                return validation_status

            # Check if column data types match the expected schema
            logger.info("Check if column data types match the expected schema")
            for col, expected_dtype in all_schema.items():
                actual_dtype = data[col].dtype
                if actual_dtype != expected_dtype:
                    logger.warning(f"Data type mismatch for column '{col}'. Expected: {expected_dtype}, Actual: {actual_dtype}")
                    validation_status = False
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")
                    return validation_status

            # If all columns and data types match, set validation_status to True
            logger.info("If all columns and data types match, set validation_status to True")
            validation_status = True
            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation status: {validation_status}")

            return validation_status

        except Exception as e:
            raise e
        
    # def calculate_data_drift(self) -> bool:

    #     data_drift_detected=None 

    #     try:
    #         reference_data = pd.read_csv(self.config.reference_data_path)
    #         current_data = pd.read_csv(self.config.unzip_data_dir)

    #         # Calculate data drift using any appropriate statistical measure (e.g., mean, standard deviation, etc.)
    #         # For example, you can use the Kolmogorov-Smirnov test to compare the distributions of numerical columns:
    #         data_drift_threshold = 0.05  # Set your desired threshold for data drift detection
    #         data_drift_detected = False

    #         for col in self.config.numerical_columns:
    #             ks_statistic, _ = ks_2samp(reference_data[col], current_data[col])
    #             if ks_statistic > data_drift_threshold:
    #                 data_drift_detected = True
    #                 logger.warning(f"Data drift detected in column '{col}'.")
    #                 with open(self.config.STATUS_FILE, 'w') as f:
    #                     f.write(f"Validation status: {data_drift_detected}")

    #         return data_drift_detected

    #     except Exception as e:
    #         raise e
        

    # def calculate_target_drift(self) -> bool:


    #     target_drift_detected=None

    #     try:
    #         reference_data = pd.read_csv(self.config.reference_data_path)
    #         current_data = pd.read_csv(self.config.unzip_data_dir)

    #         # Assuming the target variable column is named 'target':
    #         reference_target = reference_data['target']
    #         current_target = current_data['target']

    #         # Calculate target drift using any appropriate statistical measure (e.g., Kolmogorov-Smirnov test)
    #         target_drift_threshold = 0.05  # Set your desired threshold for target drift detection
    #         target_drift_detected = False

    #         ks_statistic, _ = ks_2samp(reference_target, current_target)
    #         if ks_statistic > target_drift_threshold:
    #             target_drift_detected = True
    #             logger.warning("Target drift detected.")
    #             with open(self.config.STATUS_FILE, 'w') as f:
    #                     f.write(f"Validation status: {target_drift_detected}")

    #         return target_drift_detected

    #     except Exception as e:
    #         raise e