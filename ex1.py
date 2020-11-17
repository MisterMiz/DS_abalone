import streamlit as st
import pandas as pd
import pickle

st.write("""
# Hello!   
""")

st.sidebar.header('User Input')
st.sidebar.subheader('Please enter your data:')


def get_input():
    # widget
    v_Sex = st.sidebar.radio('Sex', ['Male', 'Female', 'Infant'])
    v_Length = st.sidebar.slider('Length', 0.075000, 0.745000, 0.506790)
    v_Diameter = st.sidebar.slider('Diameter', 0.055000, 0.600000, 0.400600)
    v_Height = st.sidebar.slider('Height', 0.01, 0.24, 0.1388)
    v_Whole_weight = st.sidebar.slider('Whole_weight', 0.002, 2.55, 0.785165)
    v_Shucked_weight = st.sidebar.slider(
        'Shucked_weight', 0.001000, 1.070500, 0.308956)
    v_Viscera_weight = st.sidebar.slider(
        'Viscera_weight', 0.000500, 0.541000, 0.170249)
    v_Shell_weight = st.sidebar.slider(
        'Shell_weight', 0.001500, 1.005000, 0.249127)

    if v_Sex == 'Male':
        v_Sex = 'M'
    elif v_Sex == 'Female':
        v_Sex = 'F'
    else:
        v_Sex = 'I'

    # dictionary
    data = {'Sex': v_Sex, 'Length': v_Length, 'Diameter': v_Diameter, 'Height': v_Height, 'Whole_weight': v_Whole_weight,
            'Shucked_weight': v_Shucked_weight, 'Viscera_weight': v_Viscera_weight, 'Shell_weight': v_Shell_weight}

    # create data frame
    data_df = pd.DataFrame(data, index=[0])
    return data_df

# df = get_input()

# cat_data = pd.get_dummies(df[['Sex']])

# data_sample = pd.read_csv('abalone_sample_data.csv')
# df = pd.concat([df, data_sample],axis=0)

# X_new = pd.concat([cat_data, df], axis=1)
# X_new = X_new[:1]
# X_new = X_new.drop(columns=['Sex'])


# st.write(df)
# st.write(data_sample)
# # st.write(X_new)
# st.write(cat_data)

# load_nor = pickle.load(open('normalization.pkl', 'rb'))
# X_new = load_nor.transform(X_new)
# st.write(X_new)

# load_knn = pickle.load(open('best_knn.pkl', 'rb'))
# # Apply model for prediction
# prediction = load_knn.predict(X_new)
# st.write(prediction)

df = get_input()
# st.write(df)

data_sample = pd.read_csv('abalone_sample_data.csv')
df = pd.concat([df, data_sample],axis=0)
# st.write(df)

cat_data = pd.get_dummies(df[['Sex']])
# st.write(cat_data)

#Combine all transformed features together
X_new = pd.concat([cat_data, df], axis=1)
X_new = X_new[:1] # Select only the first row (the user input data)
#Drop un-used feature
X_new = X_new.drop(columns=['Sex'])
st.write(X_new)


# -- Reads the saved normalization model
load_nor = pickle.load(open('normalization.pkl', 'rb'))
#Apply the normalization model to new data
X_new = load_nor.transform(X_new)
st.write(X_new)

load_knn = pickle.load(open('best_knn.pkl', 'rb'))
# Apply model for prediction
prediction = load_knn.predict(X_new)
st.write(prediction)