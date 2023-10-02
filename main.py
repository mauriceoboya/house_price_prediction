import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.figure_factory as ff

st.write('BOSTON HOUSING PRICE MACHINE LEARNING PROJECT')


URL='./Housing.csv'


def load_data(nrows):
    dataset=pd.read_csv(URL,nrows=nrows)
    dataset=pd.get_dummies(columns=['mainroad','guestroom','prefarea','airconditioning','hotwaterheating','basement','furnishingstatus'],drop_first=True,data=dataset)
    return dataset


load_state_data=st.write('Loading data........')
dataset=load_data(100)
#load_state_data.write('Loading.....done!')

st.write('First 100 rows of the loaded data')
st.write(dataset)

st.write('Correlation within our dataset')
st.write('Higher correlation between the predicter variables has a negative effect on the stability of the model performance thus it is essential to check and deal with it.')
st.write(dataset.corr())

mean_price = dataset['price'].mean()
std_threshold = 3
outliers = (dataset['price'] - mean_price).abs() > std_threshold * dataset['price'].std()
dataset.loc[outliers, 'price'] = mean_price

arr = dataset['price']
fig, ax = plt.subplots()
ax.boxplot(arr)

st.pyplot(fig)