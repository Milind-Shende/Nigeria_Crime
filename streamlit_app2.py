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
from datetime import datetime
import random
from PIL import Image

data_url = 'https://github.com/Milind-Shende/Nigeria_Crime/raw/main/socio_model_prediction.csv'
real_data = pd.read_csv(data_url)



ROOT_DIR = os.getcwd()
SAVED_DIR_PATH = "artifacts"
SAVED_MODEL_FOLDER="model_trainer"
# MODEL_FILE_NAME = "model.pkl"
MODEL_FILE_NAME = "model_file.model"
TRANSFORMER_FILE_NAME="transformer.joblib"
# TARGET_ENCODER_FILE_DIR="target_encoder"
# TARGET_ENCODER_FILE_NAME="target_encoder.pkl"

MODEL_DIR = os.path.join(ROOT_DIR,SAVED_DIR_PATH,SAVED_MODEL_FOLDER,MODEL_FILE_NAME)
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
st.set_page_config(page_title="Nigeria_Crime!!!",layout="wide")

# About page
def about_page():
    # Display the title
    st.title('Predicting Terrorism & Analyzing Crime in Nigeria with ML')
    st.markdown('## **Project Background**')
    st.write('According to Wikipedia, Nigeria is considered to be a country with a high level of crime, ranking 17th among the least peaceful countries in the world, and during the first half of 2022, almost 6,000 people were killed by jihadists, kidnappers, bandits, or the Nigerian army.Being able to tackle the rate of crime in the country is a big plus to the security of the nation. The ability of the security agency to have a clear understanding of the distribution of different crimes committed and also able to anticipate/predict possible crime outbursts will go a long way to tackling the security challenges of the nation.')
    st.markdown('## **The problem**')
    st.write('The problem this project is targeted to solve is to help the security agencies to mitigate the rate of crime committed in the country by giving the security agencies reasonable insight into the distribution of crime committed in Nigeria, and also enable them to anticipate possible crime and location of the crime, in order to be able to make adequate security checks and take the necessary security measures.')

def visualization_page():
    visualization_image_url = "https://github.com/Milind-Shende/Nigeria_Crime/raw/main/Screenshot.png"
    st.image(visualization_image_url, use_column_width=True)

def fetch_multilingual_literacy(state):
    relevant_data = real_data[real_data['state'] == state]
    return relevant_data['multilingual_literacy'].values[0]

def fetch_literacy(state):
    relevant_data = real_data[real_data['state'] == state]
    return relevant_data['literacy'].values[0]

def fetch_university_admission(state):
    relevant_data = real_data[real_data['state'] == state]
    return relevant_data['university_admission'].values[0]

def fetch_sanitation(state):
    relevant_data = real_data[real_data['state'] == state]
    return relevant_data['sanitation'].values[0]

def fetch_electricity(state):
    relevant_data = real_data[real_data['state'] == state]
    return relevant_data['electricity'].values[0]

def fetch_state_unemployment(state):
    relevant_data = real_data[real_data['state'] == state]
    return relevant_data['state_unemployment'].values[0]

def fetch_avg_household_size(state):
    relevant_data = real_data[real_data['state'] == state]
    return relevant_data['avg_household_size'].values[0]

def fetch_region_population_size(state):
    relevant_data = real_data[real_data['state'] == state]
    return relevant_data['region_population_size'].values[0]

def fetch_drinking_water(state):
    relevant_data = real_data[real_data['state'] == state]
    return relevant_data['drinking_water'].values[0]
 


def generate_input_data(state, selected_date, isweekday=True, is_holiday=False):
    multilingual_literacy = fetch_multilingual_literacy(state)
    literacy = fetch_literacy(state)
    university_admission = fetch_university_admission(state)
    sanitation = fetch_sanitation(state)
    electricity = fetch_electricity(state)
    state_unemployment = fetch_state_unemployment(state)
    avg_household_size = fetch_avg_household_size(state)
    region_population_size = fetch_region_population_size(state)
    drinking_water = fetch_drinking_water(state)

    # Include the missing features
    month = selected_date.month
    day = selected_date.day
    isweekday = selected_date.weekday() < 5  # Assuming Monday to Friday are weekdays
    is_holiday = False  # You need to determine whether the selected date is a holiday or not

    input_data = {
        'State': state,
        'multilingual_literacy': multilingual_literacy,
        'literacy': literacy,
        'university_admission': university_admission,
        'sanitation': sanitation,
        'electricity': electricity,
        'state_unemployment': state_unemployment,
        'avg_household_size': avg_household_size,
        'region_population_size': region_population_size,
        'drinking_water': drinking_water,
        'month': month,
        'day': day,
        'isweekday': isweekday,
        'is_holiday': is_holiday
    }
    return input_data

def generate_future_input_data(state, selected_date, isweekday=True, is_holiday=False):
    multilingual_literacy = fetch_multilingual_literacy(state)
    literacy = fetch_literacy(state)
    university_admission = fetch_university_admission(state)
    sanitation = fetch_sanitation(state)
    electricity = fetch_electricity(state)
    state_unemployment = fetch_state_unemployment(state)
    avg_household_size = fetch_avg_household_size(state)
    region_population_size = fetch_region_population_size(state)
    drinking_water = fetch_drinking_water(state)

    # Include the missing features
    month = selected_date.month
    day = selected_date.day
    isweekday = selected_date.weekday() < 5  # Assuming Monday to Friday are weekdays
    is_holiday = False  # You need to determine whether the selected date is a holiday or not

    input_data = {
        'State': state,
        'multilingual_literacy': multilingual_literacy,
        'literacy': literacy,
        'university_admission': university_admission,
        'sanitation': sanitation,
        'electricity': electricity,
        'state_unemployment': state_unemployment,
        'avg_household_size': avg_household_size,
        'region_population_size': region_population_size,
        'drinking_water': drinking_water,
        'month': month,
        'day': day,
        'isweekday': isweekday,
        'is_holiday': is_holiday
    }
    return input_data

def prediction_page():
    # Title and input fields
    st.title('Predicting Terrorism & Analyzing Crime in Nigeria with ML')
    st.subheader(':clipboard: Information')

    # User inputs
    state_options = ['Abia', 'Abuja', 'Adamawa', 'Akwa Ibom', 'Anambra', 'Bauchi',
       'Bayelsa', 'Benue', 'Borno', 'Cross River', 'Delta', 'Ebonyi',
       'Edo', 'Ekiti', 'Enugu', 'Gombe', 'Imo', 'Jigawa', 'Kaduna',
       'Kano', 'Katsina', 'Kebbi', 'Kogi', 'Kwara', 'Lagos', 'Nasarawa',
       'Niger', 'Ogun', 'Ondo', 'Osun', 'Oyo', 'Plateau', 'Rivers',
       'Sokoto', 'Taraba', 'Yobe', 'Zamfara']
    selected_state = st.selectbox('State', state_options)
    # Define the list of valid years
    valid_years = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025,2026]


    # Convert the valid years to datetime objects for min and max values
    min_date = datetime(min(valid_years), 1, 1)
    max_date = datetime(max(valid_years), 12, 31)

    # Set a default date value within the valid range
    default_date = datetime(2026, 12, 31)  # You can choose a different default date if needed

    # Date input widget with restricted year selection
    selected_date = st.date_input('Date', default_date, min_value=min_date, max_value=max_date)

    isweekday = selected_date.weekday() < 5  # Assuming Monday to Friday are weekdays
    is_holiday = False  # You need to determine whether the selected date is a holiday or not

    # Generate input data for prediction
    input_data = generate_input_data(selected_state, selected_date, isweekday, is_holiday)

    # Prediction button
    if st.button('Predict'):
        # Prepare input data for backend processing
        perform_backend_prediction(input_data, selected_date)



def perform_backend_prediction(input_data, selected_date, threshold=0.5):
    try:
        multilingual_literacy = float(input_data['multilingual_literacy'])
        literacy = float(input_data['literacy'])
        university_admission = float(input_data['university_admission'])
        sanitation = float(input_data['sanitation'])
        electricity = float(input_data['electricity'])
        state_unemployment = float(input_data['state_unemployment'])
        avg_household_size = float(input_data['avg_household_size'])
        region_population_size = float(input_data['region_population_size'])
        drinking_water = float(input_data['drinking_water'])

        # Extract month, day, isweekday, and is_holiday from the selected_date
        month = selected_date.month
        day = selected_date.day
        isweekday = selected_date.weekday() < 5  # Assuming Monday to Friday are weekdays
        is_holiday = False  # You need to determine whether the selected date is a holiday or not

        # Combine user inputs with other default features
        input_data.update({
            'multilingual_literacy': multilingual_literacy,
            'literacy': literacy,
            'university_admission': university_admission,
            'sanitation': sanitation,
            'electricity': electricity,
            'state_unemployment': state_unemployment,
            'avg_household_size': avg_household_size,
            'region_population_size': region_population_size,
            'drinking_water': drinking_water,
            'month': month,
            'day': day,
            'isweekday': isweekday,
            'is_holiday': is_holiday,
            'Date': selected_date  # Add the selected_date to the input_data dictionary
        })
        # Convert input data to a Pandas DataFrame
        input_df = pd.DataFrame(input_data, index=[0])

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
        predicted_prob = loaded_model.predict(dmatrix)  # Predicted probability
        predicted_class = 1 if predicted_prob >= threshold else 0

        # Calculate the predicted probability in percentage
        pred_prob_percentage = predicted_prob[0] * 100

        # Print the prediction result in the desired format
        state = input_data['State']
        date_to_check = input_data['Date'].strftime('%Y-%m-%d')
        prediction_text = "YES" if predicted_class == 1 else "NO"
        st.write(f'Probability of an attack in the state of {state} on {date_to_check}: {int(round(pred_prob_percentage))}%')




    except Exception as e:
        # error message if an exception occurs
        st.error(f"An error occurred: {e}")

# Teams page
def collaborators_page():
    st.title('Predicting Terrorism & Analyzing Crime in Nigeria with ML')
    st.write("I'm writing to express my heartfelt appreciation for each one of you who have contributed to our project,\"Predicting Terrorism & Analyzing Crime in Nigeria with ML\".Your dedication, expertise,and hard work have been pivotal in bringing this project to fruition,even while we're physically separated by distance.Your valuable insights and unwavering commitment have made a lasting impact, and I'm truly inspired by our collective achievements.I'm truly grateful to work with such a talented and dedicated group of collaborators.")
    st.write("Warm regards,")

    st.write(':male-scientist: Meet our awesome team members:')
    st.write("-Milind Shende")
    st.write('-Umesh Patil')
    st.write('-Miho Rosenberg')
    st.write('-Abomaye Eniatorudabo')
    st.write('-Anjali Dashora')
    st.write('-Robson Serafim')
    st.write('-Indrajith')
    # st.write('-Samuel David Egwu')
    st.write('-Devyash Jain')
    # st.write('-Danish Mehmood')
    st.write('-Devyash Jain')
    st.write('-Walid hossain')
    # st.write('-Samson Oni')
    # st.write('-Shivanshi Arora')
    # st.write('-Shreya chawla')
    st.write('-Oluchukwu')
    # st.write('-Ololade Ogunleye')
    # st.write('-Hannah Marie Pacis')
    # st.write('-Nofisat Hamod')
    # st.write('-Richard oveh')
    # st.write('-Touib Ogunremi')
    # st.write('-Sulagna Parida')
    # st.write('-Vishnu Pandey')

def nigeria_image():
    # Load and display the Nigerian flag image
    nigerian_flag_image_url = "https://raw.githubusercontent.com/Milind-Shende/Nigeria_Crime/main/omdena-nigeria.png"
    image_width = 150
    st.sidebar.image(nigerian_flag_image_url, width=image_width)


def tools_section():
    st.sidebar.title('Tools Used')
    # Create a dictionary of tools with their names and links
    tools = {
        'Python 🔗': 'https://www.python.org/',
        'Pandas 🔗': 'https://pandas.pydata.org/',
        'Numpy 🔗': 'https://numpy.org/',
        'Matplotlib 🔗': 'https://matplotlib.org/',
        'seaborn 🔗': 'https://seaborn.pydata.org/',
        'Pygwalker 🔗': 'https://docs.kanaries.net/pygwalker',
        'XGBoost 🔗': 'https://xgboost.readthedocs.io/en/latest/index.html',
        'scikit-learn 🔗': 'https://scikit-learn.org/stable/index.html',
        'Streamlit 🔗': 'https://streamlit.io/',
        # Add more tools with their links as needed
    }
    
    # Display tools with links using Markdown syntax
    for tool, link in tools.items():
        st.sidebar.markdown(f"- {tool.replace('🔗', '<a href=' + link + ' target=_blank>🔗</a>')} ", unsafe_allow_html=True)




    # Add more team members as needed

# Create a dictionary with page names and their corresponding functions
pages = {
    'About': about_page,
    'Visualization ':visualization_page, 
    'Prediction': prediction_page,
    'Collaborators': collaborators_page,
}

# Streamlit application
def main():
    # Sidebar navigation
    st.sidebar.title('Navigation')
    selected_page = st.sidebar.radio('Go to', list(pages.keys()))
    nigeria_image()
    tools_section()
    

    # Display the selected page content
    pages[selected_page]()

# Run the Streamlit application
if __name__ == '__main__':
    main()