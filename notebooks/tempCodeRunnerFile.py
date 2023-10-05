import streamlit as st
import pickle
import sklearn
import pandas as pd
import numpy as np

model = pickle.load(open('ds-salary-predictor-1.sav', 'rb'))

st.title('Data Science Salary Prediction')

data = pd.read_csv('..\data\processed\modeling_used.csv')

valCol = {}
for col in data.columns.drop('salary_in_usd'):
    valCol.update({
        col: tuple(data[col].unique())
    })

print(valCol)
