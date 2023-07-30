import numpy as np
import pickle
import gradio as gr
import pandas as pd


# loading the saved model
loaded_model = pickle.load(open('trained_gbrmodel.sav', 'rb'))
print(loaded_model)

# creating a function for prediction
# creating a function for prediction


def biomass_prediction(*inputs):
    """
      Predicts five properties of biomass pyrolysis products based on seven input features of the biomass.

      Parameters:
      input1 (float): Lignin content of the biomass.
      input2 (float): Ash content of the biomass.
      input3 (float): O% content of the biomass.
      input4 (float): H% content of the biomass.
      input5 (float): N% content of the biomass.
      input6 (float): Particle size of the biomass.
      input7 (float): Pyrolysis temperature.

      Returns:
      A tuple of five floats representing predicted properties of the biomass pyrolysis products:
      - Biomass H/C
      - Biomass O/C
      - Biomass Oil_yield
      - Biomass Gas_yield
      - Biomass Char_yield
    """
    # check if the cells are all zeros
    # concatenate the inputs into a numpy array
    input_data = pd.DataFrame(
        [inputs], columns=['Lig', 'Ash%', 'O-%', 'H-%', 'N-%', 'Size', 'PT'])

    # check if all values are zero or negative
    if np.all(input_data <= 0):
        return 0, 0, 0, 0, 0
    else:
      # make predictions using the loaded machine learning model
        prediction = loaded_model.predict(input_data)
        prediction = np.round(prediction, 2)
        # extract each prediction value and save it to a separate variable
        output1 = prediction[0][0]
        output2 = prediction[0][1]
        output3 = str(prediction[0][2])+" wt%"
        output4 = str(prediction[0][3]) + " wt%"
        output5 = str(prediction[0][4]) + " wt%"
        # return a tuple of predicted output values

        return output1, output2, output3, output4, output5


def create_gui():
    """
      Creates and launches the Gradio user interface for the biomass pyrolysis prediction.
    """
    # define the input and output components of the GUI
    inputs = [
        gr.components.Number(label="Lignin Content",
                             info="Please enter the lignin content"),
        gr.components.Number(label="Ash Content",
                             info="Please enter the ash content"),
        gr.components.Number(label="O% Content",
                             info="Please enter the O% content"),
        gr.components.Number(label="H% Content", info="Please the H% content"),
        gr.components.Number(label="N% Content",
                             info="Please enter the N% content"),
        gr.components.Number(label="Biomass Particle size",
                             info="Please enter the particle size of the biomass"),
        gr.components.Number(label="Pyrolysis Temeprature",
                             info="Please enter the Pyrolysis temperature"),
    ]
    outputs = [
        gr.components.Textbox(label="Biomass H/C"),
        gr.components.Textbox(label="Biomass O/C"),
        gr.components.Textbox(label="Biomass Oil_yield"),
        gr.components.Textbox(label="Biomass Gas_yield"),
        gr.components.Textbox(label="Biomass Char_yield"),
    ]
    # define the title and description of the GUI
    title = "GUI for Biomass Pyrolysis Prediction"
    description = "This GUI uses machine learning to predict the yield of bio-oil, biochar, biogas, H/C and O/C produced during biomass pyrolysis. The app takes seven features of the biomass, such as ash content, lignin content, pyrolysis temperature, and predicts five properties of the pyrolysis products, including bio-oil yield, biochar yield, and biogas yield. This prediction can help optimize the pyrolysis process, improve bioenergy production efficiency, and reduce the environmental impact of biomass conversion."

    gr.Interface(fn=biomass_prediction, inputs=inputs, outputs=outputs,
                 title=title, description=description).launch(share=True)


if __name__ == '__main__':

    # loading the saved model
    loaded_model = pickle.load(open('trained_gbrmodel.sav', 'rb'))
    # create gui
    create_gui()
