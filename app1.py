import streamlit as st
import pickle
import pandas as pd
import numpy as np


# page view
def page_view():
    # title
    st.markdown("<h1 style='text-align: center;'>Artificial Intelligence (AI) Job Roles Salary Prediction</h1>", unsafe_allow_html=True)
    # line spacing with horizontal straight line
    st.markdown('***')
    # team
    st.header(":technologist: Contributors:")
    team="""
    1. <a href="https://github.com/aridiawan">Ari Adhi Hermawan</a>
    2. <a href="https://github.com/fnkhairudin">Faisal Nur Khairudin</a>
    """
    st.markdown(team, unsafe_allow_html=True)

    # notes
    st.header(":spiral_note_pad: Notes:")
    note ="""1. Limitation:
    - This model is for educational purposes only and demonstrate how machine learning is works. Not for production phase.
    - This dataset has been taken from 2020-2023 (Latest downloaded: September 6, 2023 at 09:58 AM on this <a href="https://ai-jobs.net/salaries/download/">link!</a>). Therefore, if you use this model in 3-5+ years later to predict how much your salary will be, maybe it's not accurate anymore.
2. Model:
    - The model used is XGBoost (Extreme Gradient Boosting)
    - The prediction results can be off by approximately US$ 38897 (MAE) from the actual salary, or in percentage terms the salary prediction will be off by approximately 31.82% (MAPE) from the actual salary.
3. User Input
    - employee_residence, company_location, and company_region input are based on <a href="https://github.com/fnkhairudin/Data-Scientist-Salary-Prediction/blob/main/data/raw/ISO-3166-Countries-with-Regional-Codes.csv">ISO 3166 countries with regional code</a>.
    - abroad: whether or not the employer's main office or contracting branch is located in the same employee's primary country of residence.
    """
    st.markdown(note, unsafe_allow_html=True)

# load model
def load_model():
    file_name = "models/ds-salary-predictor-1.1.sav"
    with open(file_name, 'rb') as pickled:
        model = pickle.load(pickled)

    return model

model = load_model()

# load dataset
def load_data():
    file_name = "data/processed/modeling_used_1.1.csv"
    with open(file_name, 'rb') as data:
        data = pd.read_csv(data)

    return data

data = load_data()


# FUNCTION
def user_report():
    # collect the columns and its values
    valCol = {}
    cols = data.drop(columns='salary_in_usd').columns
    for col in cols:
        valCol.update({
            col: tuple(data[col].unique())
        })
        
    # define the user input
    experience_level = st.sidebar.selectbox(cols[0], ["Entry-level / Junior", "Mid-level / Intermediate", 'Senior-level / Expert', 'Executive-level / Director'])
    employment_type = st.sidebar.selectbox(cols[1], ['Part-time', 'Full-time', 'Contract', 'Freelance'])
    employee_residence = st.sidebar.selectbox(cols[2], valCol[cols[2]])
    remote_ratio = st.sidebar.selectbox(cols[3], ['No remote work (less than 20%)', 'Partially remote/hybird', 'Fully remote (more than 80%)'])
    company_location = st.sidebar.selectbox(cols[4], valCol[cols[4]])
    company_size = st.sidebar.selectbox(cols[5], ['less than 50 employees (small)', '50 to 250 employees (medium)', 'more than 250 employees (large)'])
    job_position = st.sidebar.selectbox(cols[6], valCol[cols[6]])
    job_scope = st.sidebar.selectbox(cols[7], valCol[cols[7]])
    company_region = st.sidebar.selectbox(cols[8], valCol[cols[8]])
    aboard = st.sidebar.selectbox(cols[9], valCol[cols[9]])

    # experience level
    if experience_level == 'Entry-level / Junior':
        experience_level = 'EN'
    elif experience_level == 'Mid-level / Intermediate':
        experience_level = 'MI'
    elif experience_level == 'Senior-level / Expert':
        experience_level = 'SE'
    else:
        experience_level = 'EX'

    # employment type
    if employment_type == 'Part-time':
        employment_type = 'PT'
    elif employment_type == 'Full-time':
        employment_type = 'FT'
    elif employment_type == 'Contract':
        employment_type = 'CT'
    else:
        employment_type = 'FL'

    # remote ratio
    if remote_ratio == 'No remote work (less than 20%)':
        remote_ratio = 0
    elif remote_ratio == 'Partially remote/hybird':
        remote_ratio = 50
    else:
        remote_ratio = 100
    
    # company size
    if company_size == 'less than 50 employees (small)':
        company_size = 'S'
    elif company_size == '50 to 250 employees (medium)':
        company_size = 'M'
    else:
        company_size = 'L'
    
    # user report
    user_report_data = {
        cols[0]:experience_level,
        cols[1]:employment_type,
        cols[2]:employee_residence,
        cols[3]:remote_ratio,
        cols[4]:company_location,
        cols[5]:company_size,
        cols[6]:job_position,
        cols[7]:job_scope,
        cols[8]:company_region,
        cols[9]:aboard,
    }
    # return as DataFrame
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data

def main():
    # page_view
    page_view()

    # collect the user input as tabular format
    user_data = user_report()
    st.header('Your Data')
    st.write(user_data)

    # predict the salary
    st.subheader('Salary Prediction')
    if st.button('Predict your salary here!'):
        salary = model.predict(user_data)
        st.subheader('$'+str(np.round(salary[0], 2)))

if __name__  == "__main__":
    main()