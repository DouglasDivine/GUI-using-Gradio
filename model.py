import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Import basic operations and plotting
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
# Import error performance measure, preprocessing etc. from sklearn
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def load_data():
    data = pd.read_csv('clean_data.xlsx')
    # drop duplicates
    data = data.drop_duplicates()
    data.reset_index(drop=True, inplace=True)
    data = data.astype('float32')

    # split into input (X) and output (Y) variables
    X = data[['Cel', 'Hem', 'Lig', 'Vm%', 'Ash%', 'FC%', 'C-%',
              'H-%', 'O-%', 'N-%', 'Size', 'HR', 'PT', 'Temp']]

    #X = data[['Lig', 'Ash%', 'O-%', 'H-%', 'N-%', 'Size', 'PT']]
    Y = data[['H/C', 'O/C', 'Oil_yield%', 'Gas_yield%', 'Char_yield%']]

    return X, Y


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    A function that replaces the nan or missing values of each column with the mean value of that column respectively,
    converts the data to the right format
    """
    for col in data.columns:
        if data[col].dtype != 'float':
            data[col] = pd.to_numeric(data[col], errors='coerce')
    return data.fillna(data.mean())


def outlier_threshold(normality, k=1.5):
    # use k =1.5
    q1 = np.quantile(normality, 0.25)
    q3 = np.quantile(normality, 0.75)
    threshold = q1 - k*(q3-q1)
    return threshold


def remove_outlier(X):
    clf = IsolationForest(
        n_estimators=100,
        max_samples='auto',
        n_jobs=-1,
        random_state=5)

    clf.fit(X)
    normality_df = pd.DataFrame(
        clf.decision_function(X), columns=['normality'])
    threshold = outlier_threshold(normality_df['normality'].values, k=1.5)
    # Plots the distribution and the threshold
    fig = px.histogram(normality_df, x='normality', width=400, height=400)
    fig.add_vline(x=threshold, line_width=3,
                  line_dash="dash", line_color="red")
    fig.update_layout(width=670, height=400)
    fig.show()

    # remove outliers from both the x and y data
    x_new = X[normality_df['normality'].values >= threshold]
    y_new = Y[normality_df['normality'].values >= threshold]
    print('{} out of {} observations are removed from the dataset'.format(
        (X.shape[0] - x_new.shape[0]), X.shape[0]))

    return x_new, y_new


def build_model(x, y, test_size=0.15, random_state=42):
    cols = [2, 4, 8, 7, 9, 10, 12]

    #x_sel = x[:, cols]
    x_sel = x[['Lig', 'Ash%', 'O-%', 'H-%', 'N-%', 'Size', 'PT']]

    x_train, x_test, y_train, y_test = train_test_split(
        x_sel, y, test_size=test_size, random_state=random_state)
    model = MultiOutputRegressor(GradientBoostingRegressor())
    model.fit(x_train, y_train)
    # Evaluate model's performance
    R2_train = np.round(model.score(x_train, y_train), 2)
    R2_test = np.round(model.score(x_test, y_test), 2)

    print(f'R2 of train set is: {R2_train}')

    print(f'R2 for test test is: {R2_test}')

    # Download the trained model
    filename = 'trained_gbrmodel.sav'
    pickle.dump(model, open(filename, 'wb'))


if __name__ == '__main__':
    # Load data
    X, Y = load_data()
    # Remove outliers from data
    x, y = remove_outlier(X)
    # train the model
    build_model(x, y)
