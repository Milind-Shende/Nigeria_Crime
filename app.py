import streamlit as st
import pickle
import os
import pandas as pd
import numpy as np
import joblib 

ROOT_DIR = os.getcwd()
SAVED_DIR_PATH = "artifacts"
SAVED_MODEL_FOLDER="model_trainer"
MODEL_FILE_NAME = "model.joblib"
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
model=joblib.load(open(MODEL_DIR,"rb"))
# print(model)
transfomer=joblib.load(open(TRANSFORMER_DIR,"rb"))
# print(transfomer)



# About page
def about_page():
    st.title('Predicting the Financial Burden of Lung Cancer')
    st.write('The project aims to develop a predictive model that estimates the annual out-of-pocket costs for patients diagnosed with Stage 3&4 lung cancer. By considering factors such as age, comorbidities, and primary insurance, the model will enable patients to proactively plan for future financial burdens associated with their diagnosis. The ultimate goal is to alleviate the financial stress and reduce the likelihood of personal bankruptcy that over 40% of cancer patients experience within four years of diagnosis.')
    
def visualization_page():...


# Main prediction page
def prediction_page():
    # Title and input fields
    st.title('Predicting the Financial Burden of Lung Cancer')
    st.subheader('Patient Information')
    year = st.text_input('Year',value='0')
    month = st.selectbox('Month', ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'))
    day = st.selectbox('Day', ('Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'))
    extended = st.selectbox('Extended', ('0', '1'))
    state = st.selectbox('State', ('Lagos', 'Kaduna', 'Unknown', 'Katsina', 'Zamfara', 'Kano',
                                    'Akwa Ibom', 'Edo', 'Taraba', 'Ondo', 'Bayelsa', 'Cross River',
                                    'Abuja', 'Oyo', 'Anambra', 'Rivers', 'Osun', 'Ekiti', 'Delta',
                                    'Enugu', 'Ogun', 'Plateau', 'Kwara', 'Imo', 'Kogi', 'Borno',
                                    'Bauchi', 'Sokoto', 'Abia', 'Benue', 'Ebonyi', 'Adamawa', 'Gombe',
                                    'Niger', 'Kebbi', 'Yobe', 'Jigawa', 'Nasarawa'))
    city = st.text_input('city')
    
    target_type = st.selectbox('target_type', ('Government (General)', 'Government (Diplomatic)',
                                                            'Educational Institution', 'Journalists & Media',
                                                            'Private Citizens & Property', 'Police',
                                                            'Religious Figures/Institutions', 'Business', 'Maritime',
                                                            'Military', 'Unknown', 'Transportation', 'Utilities',
                                                            'Violent Political Party', 'Airports & Aircraft',
                                                            'Telecommunication', 'NGO', 'Other',
                                                            'Terrorists/Non-State Militia'))
    
    nationality = st.selectbox('nationality Year', ('Nigeria', 'Great Britain', 'Libya', 'Saudi Arabia', 'Australia',
                                                    'France', 'Unknown', 'Netherlands', 'International',
                                                    'United States', 'Multinational', 'Belize', 'Germany', 'Italy',
                                                    'South Korea', 'Norway', 'China', 'Philippines', 'Croatia',
                                                    'Lebanon', 'Turkey', 'Indonesia', 'India', 'Russia', 'Poland',
                                                    'Asian', 'Syria', 'Afghanistan', 'Israel', 'Denmark', 'Namibia',
                                                    'Chad', 'South Africa', 'North Korea', 'Niger', 'Greece','Liberia'))
    
    weapon_type =st.selectbox('weapon_type',('Firearms', 'Unknown', 'Incendiary', 'Melee', 'Explosives','Chemical', 'Sabotage Equipment'))
    
    
     
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
            # st.write("Input Data Shape:", input_df.shape)
            # Convert input data to a Pandas DataFrame
            input_df = pd.DataFrame(input_data)   
            # Perform the transformation using the loaded transformer
            transformed_data = transfomer.transform(input_df)
            # st.write("Transformed Data Shape:", transformed_data.shape)
            # Reshape the transformed data as a NumPy array
            input_arr = np.array(transformed_data)
            # Make the prediction using the loaded model
            prediction = model.predict(input_arr)
            st.subheader('Prediction')
            st.write(f'The predicted total charge is: {prediction[0]}')
        except Exception as e:
        # error message if an exception occurs
            st.error(f"An error occurred: {e}")

# Teams page
def collaborators_page():
    st.title('Predicting the Financial Burden of Lung Cancer')
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