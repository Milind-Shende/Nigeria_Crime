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

ROOT_DIR = os.getcwd()
SAVED_DIR_PATH = "artifacts"
SAVED_MODEL_FOLDER="model_trainer"
MODEL_FILE_NAME = "model.joblib"
TRANSFORMER_FILE_NAME="transformer.joblib"
# TARGET_ENCODER_FILE_DIR="target_encoder"
# TARGET_ENCODER_FILE_NAME="target_encoder.pkl"

MODEL_DIR = os.path.join(ROOT_DIR, SAVED_DIR_PATH,SAVED_MODEL_FOLDER,MODEL_FILE_NAME)
# print("MODEL_PATH:-",MODEL_DIR)

TRANSFORMER_DIR= os.path.join(ROOT_DIR, SAVED_DIR_PATH,SAVED_MODEL_FOLDER,TRANSFORMER_FILE_NAME)
# print("TRANSFORMER_PATH:-",TRANSFORMER_DIR)

# TARGET_ENCODER_DIR= os.path.join(ROOT_DIR, SAVED_DIR_PATH,SAVED_ZERO_FILE,TARGET_ENCODER_FILE_DIR,TARGET_ENCODER_FILE_NAME)
# print("TARGET_ENCODER_PATH:-",TARGET_ENCODER_DIR)

# Load the Model.pkl, Transformer.pkl and Target.pkl
model=joblib.load(open(MODEL_DIR,"rb"))
# print(model)
transfomer=joblib.load(open(TRANSFORMER_DIR,"rb"))
# print(transfomer)

# Adjust the width of the Streamlit page
st.set_page_config(
    page_title="Visualisation of Crime in Nigeria",
    layout="wide")

# About page
def about_page():
    st.title('Predicting Terrorism & Analyzing Crime in Nigeria with ML')
    st.write('The problem this project is targeted to solve is to help the security agencies to mitigate the rate of crime committed in the country by giving the security agencies reasonable insight into the distribution of crime committed in Nigeria, and also enable them to anticipate possible crime and location of the crime, in order to be able to make adequate security checks and take the necessary security measures.')
    

def load_data(data_file):
    if data_file is not None:
        df = pd.read_csv(data_file)
        return df
    return None

def visualization_page():
    df=pd.read_csv("terrorism_cleaned.csv")
    
    # Add Title
    st.title("Visualisation of Crime in Nigeria")
    st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)
    # Generate the HTML using Pygwalker
    pyg_html = pyg.walk(df, return_html=True)
    # Embed the HTML into the Streamlit app
    components.html(pyg_html, height=1000, scrolling=True)

Target_labels = ['Assassination', 'Unknown', 'Facility/Infrastructure Attack',
                'Unarmed Assault', 'Armed Assault', 'Bombing/Explosion',
                'Hostage Taking (Kidnapping)',
                'Hostage Taking (Barricade Incident)', 'Hijacking']
    
# Main prediction page
def prediction_page():
    # Title and input fields
    st.title('Predicting Terrorism & Analyzing Crime in Nigeria with ML')
    st.subheader('Information')
    year = st.selectbox('Year',('1976', '1980', '1983', '1988', '1991', '1992', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021'))
    month = st.selectbox('Month', ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'))
    day = st.selectbox('Day', ('Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'))
    extended = st.selectbox('Extended', ('0', '1'))
    state = st.text_input('State')
    city = st.text_input('City')
    target_type = st.selectbox('Target Type',('Government (General)', 'Government (Diplomatic)',
                                'Educational Institution', 'Journalists & Media',
                                'Private Citizens & Property', 'Police',
                                'Religious Figures/Institutions', 'Business', 'Maritime',
                                'Military', 'Unknown', 'Transportation', 'Utilities',
                                'Violent Political Party', 'Airports & Aircraft',
                                'Telecommunication', 'NGO', 'Other',
                                'Terrorists/Non-State Militia'))
    nationality = st.text_input('Nationality')
    weapon_type =st.selectbox('Weapon Type',('Firearms', 'Unknown', 'Incendiary', 'Melee', 'Explosives','Chemical', 'Sabotage Equipment'))
    
    

    # Prediction button
    if st.button('Predict'):
        # Preprocess the input features
        try:
            input_data = {
                            'year': [year],
                            'month': [month],
                            'day': [day],
                            'extended': [extended],
                            'state': [state],
                            'city': [city],
                            'target_type': [target_type],
                            'nationality': [nationality],
                            'weapon_type': [weapon_type],
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

            # Make the prediction using the loaded model
            predicted_index = model.predict(input_arr)[0]
            predicted_attack_type = Target_labels[predicted_index]
            st.subheader('Prediction')
            st.write(f'The predicted attack type is: {predicted_attack_type}')
                
        except Exception as e:
        # error message if an exception occurs
            st.error(f"An error occurred: {e}")

# Teams page
def collaborators_page():
    st.title('Predicting Terrorism & Analyzing Crime in Nigeria with ML')
    st.write('Meet our awesome team members:')
    st.write('- Team Member 1')
    st.write('- Team Member 2')
    st.write('- Team Member 3')
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

    # Display the selected page content
    pages[selected_page]()

# Run the Streamlit application
if __name__ == '__main__':
    main()