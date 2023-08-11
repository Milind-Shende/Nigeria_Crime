import pickle
import os
import pandas as pd
import numpy as np
import joblib 
from NigeriaMLflow import logger
import pygwalker as pyg
import pandas as pd
import streamlit.components.v1 as components
import streamlit.components.v1 as stc 
import streamlit as st
from scipy.sparse import issparse
import xgboost as xgb
import pandas as pd

ROOT_DIR = os.getcwd()
SAVED_DIR_PATH = "artifacts"
SAVED_MODEL_FOLDER="model_trainer"
MODEL_FILE_NAME = "model_file.model"
TRANSFORMER_FILE_NAME="transformer.joblib"
# TARGET_ENCODER_FILE_DIR="target_encoder"
# TARGET_ENCODER_FILE_NAME="target_encoder.pkl"

MODEL_DIR = os.path.join(ROOT_DIR,MODEL_FILE_NAME)
# print("MODEL_PATH:-",MODEL_DIR)

TRANSFORMER_DIR= os.path.join(ROOT_DIR,SAVED_DIR_PATH,SAVED_MODEL_FOLDER,TRANSFORMER_FILE_NAME)
# print("TRANSFORMER_PATH:-",TRANSFORMER_DIR)

# TARGET_ENCODER_DIR= os.path.join(ROOT_DIR, SAVED_DIR_PATH,SAVED_ZERO_FILE,TARGET_ENCODER_FILE_DIR,TARGET_ENCODER_FILE_NAME)
# print("TARGET_ENCODER_PATH:-",TARGET_ENCODER_DIR)

# Load the Model.pkl, Transformer.pkl and Target.pkl
# model=joblib.load(open(MODEL_DIR,"rb"))
# Load the saved model
loaded_model = xgb.Booster(model_file=MODEL_DIR)
# print(model)
transfomer=joblib.load(open(TRANSFORMER_DIR,"rb"))
# print(transfomer)

# Adjust the width of the Streamlit page
st.set_page_config(page_title="Nigeria_Crime!!!", page_icon=":bar_chart:",layout="wide")

# About page
def about_page():
    st.title('Predicting Terrorism & Analyzing Crime in Nigeria with ML')
    st.write('The problem this project is targeted to solve is to help the security agencies to mitigate the rate of crime committed in the country by giving the security agencies reasonable insight into the distribution of crime committed in Nigeria, and also enable them to anticipate possible crime and location of the crime, in order to be able to make adequate security checks and take the necessary security measures.')
    
    
# Main prediction page
def prediction_page():
    # Title and input fields
    st.title('Predicting Terrorism & Analyzing Crime in Nigeria with ML')
    st.subheader('Information')
    State = st.text_input('State')
    multilingual_literacy = st.text_input('multilingual_literacy')
    literacy = st.text_input('literacy')
    university_admission = st.text_input('university_admission')
    sanitation = st.text_input('sanitation')
    electricity = st.text_input('electricity')
    state_unemployment = st.text_input('state_unemployment')
    avg_household_size = st.text_input('avg_household_size')
    region_population_size =st.text_input('region_population_size')
    drinking_water =st.text_input('drinking_water')
    month =st.text_input('month')
    day =st.text_input('day')
    isweekday =st.text_input('isweekday')
    is_holiday =st.text_input('is_holiday')

    
    

    # Prediction button
    if st.button('Predict'):
        # Preprocess the input features
        try:
            input_data = {
                            'State': [State],
                            'multilingual_literacy': [multilingual_literacy],
                            'literacy': [literacy],
                            'university_admission': [university_admission],
                            'sanitation': [sanitation],
                            'electricity': [electricity],
                            'state_unemployment': [state_unemployment],
                            'avg_household_size': [avg_household_size],
                            'region_population_size': [region_population_size],
                            'drinking_water':[drinking_water],
                            'month':[month],
                            'day':[day],
                            'isweekday':[isweekday],
                            'is_holiday':[is_holiday]
                        }
            
            # Convert input data to a Pandas DataFrame
            input_df = pd.DataFrame(input_data)

            # Perform the transformation using the loaded transformer
            transformed_data = transfomer.transform(input_df)

            # Convert transformed_data to a dense NumPy array
            if issparse(transformed_data):
                input_arr = transformed_data.toarray()
            else:
                input_arr = np.array(transformed_data)

            # Convert input array to a DMatrix
            dmatrix = xgb.DMatrix(input_arr)

            # Make the prediction using the loaded model
            predicted_prob = loaded_model.predict(dmatrix)
            predicted_class = 1 if predicted_prob >= 0.5 else 0

            st.subheader('Prediction')
            prediction_text = "YES" if predicted_class == 1 else "NO"
            st.write(f'The predicted class is: {prediction_text}')
                
        except Exception as e:
        # error message if an exception occurs
            st.error(f"An error occurred: {e}")

# Teams page
def collaborators_page():
    st.title('Predicting Terrorism & Analyzing Crime in Nigeria with ML')
    st.write("I'm writing to express my heartfelt appreciation for each one of you who have contributed to our project,\"Predicting Terrorism & Analyzing Crime in Nigeria with ML\".Your dedication, expertise,and hard work have been pivotal in bringing this project to fruition,even while we're physically separated by distance.Your valuable insights and unwavering commitment have made a lasting impact, and I'm truly inspired by our collective achievements.I'm truly grateful to work with such a talented and dedicated group of collaborators.")
    st.write("Warm regards,")

    st.write('Meet our awesome team members:')
    st.write('-Umesh')
    st.write("-Milind Shende")
    st.write('-Miho Rosenberg')
    st.write('-Abomaye Eniatorudabo')
    st.write('-Robson Serafim')
    st.write('-Indrajith')
    st.write('-Anjali Dashora')
    st.write('-Samuel David Egwu')
    st.write('-Devyash Jain')
    st.write('-Danish Mehmood')
    st.write('-Devyash Jain')
    st.write('-Walid hossain')
    st.write('-Samson Oni')
    st.write('-Shivanshi Arora')
    st.write('-Shreya chawla')
    st.write('-Oluchukwu')
    st.write('-Ololade Ogunleye')
    st.write('-Hannah Marie Pacis')
    st.write('-Nofisat Hamod')
    st.write('-Richard oveh')
    st.write('-Touib Ogunremi')
    st.write('-Sulagna Parida')
    st.write('-Vishnu Pandey')


    # Add more team members as needed

# Create a dictionary with page names and their corresponding functions
pages = {
    'About': about_page,
    'Prediction': prediction_page,
    'Collaborators': collaborators_page,
}

# Streamlit application
def main():
    # Sidebar navigation
    st.sidebar.title('Navigation')
    selected_page = st.sidebar.radio('Go to', list(pages.keys()))

    # Display the selected page content
    pages[selected_page]()

# Run the Streamlit application
if __name__ == '__main__':
    main()