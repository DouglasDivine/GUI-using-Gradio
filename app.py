import numpy as np
import pickle
import streamlit as st
import pandas as pd


# loading the saved model
loaded_model = pickle.load(open('trained_gbrmodel.sav', 'rb'))
print(loaded_model)

# creating a function for prediction


def biomass_prediction(input_data):

    # changing the input_data to numpy array
    input_data_as_np = np.asarray(input_data)

    print(input_data_as_np)

    # reshape the array as we are prediciting for one instance
    input_data_reshaped = input_data_as_np.reshape(1, -1)
    print(input_data_reshaped)

    prediction = loaded_model(input_data_reshaped)
    print(prediction)

    return prediction


def main():
    # Giving a title
    st.title('Biomass prediction Web App')

    # Getting the input data from the user
    [['Lig', 'Ash%', 'O-%', 'H-%', 'N-%', 'Size', 'PT']]
    Lig = st.text_input('Lignin Content')
    Ash = st.text_input('Ash Content')
    O = st.text_input('O% Content')
    H = st.text_input('H% Content')
    N = st.text_input('N% Content')
    Size = st.text_input('Biomass Particle Size')
    Pt = st.text_input('Pyrolysis Temperature')

    # Code for prediction
    output = ''
    # creating a button for prediction
    if st.button('Biomass H/C', 'O/C', 'Oil_yield%', 'Gas_yield%', 'Char_yield%'):
        output = biomass_prediction([Lig, Ash, O, H, N, Size, Pt])

    st.success(output)


if __name__ == '__main__':
    main()
