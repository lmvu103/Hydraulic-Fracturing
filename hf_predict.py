import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import missingno as msno
import seaborn as sns
import scipy as sp
import altair as alt
from vega_datasets import data
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import validation_curve, GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree

st.title("***HYDRAULIC FRACTURING PREDICTION***")

# This is the ready-to-use version of that data already converted to CSV.
# Load this instead to skip pre-processing steps above.
url = 'https://raw.githubusercontent.com/yohanesnuwara/energy-analysis/main/data/SPE_ML_data.csv'


#wells = pd.read_csv(url)
#st.header("Raw Data")
#st.write(wells)

#df = wells['# Clusters per Stage']
#st.subheader("Filter Data")
#st.write(df)

def load_data():
    data = pd.read_csv(url)
    return data


def sort_values(df, column_to_sort, ascending=True):
    sorted_df = df.sort_values(column_to_sort, ascending=ascending)
    sorted_df = sorted_df[['Lease', column_to_sort]]
    return sorted_df


df = load_data()

sorted_df = sort_values(df, '# Clusters per Stage', ascending=False)

st.write(df)


#st.write(sorted_df)


def t_test(N, alpha=0.05):
    # Default 5% confidence level
    t = sp.stats.t.isf(alpha / 2, N - 2)
    r_crit = t / np.sqrt((N - 2) + np.power(t, 2))
    return r_crit


# t-test
N = len(df)  # number of leases
r_crit = t_test(N)

# Drop uncorrelated features
uncorrelated_features = [' Wellbore Diameter (ft) ', 'Porosity', ' Water Saturation ',
                         'Sep. Temperature (deg F)', 'Sep. Pressure (psi)',
                         '# Stages', '# Clusters ', ' Sandface Temp (deg F) ',
                         ' Static Wellhead Temp (deg F) ', 'Condensate Gravity (API)',
                         ' H2S ', '# of Total Proppant (Lbs)']

wells = df.drop(uncorrelated_features, axis=1)

# Encode formation column
le = LabelEncoder()
wells['Formation/Reservoir'] = le.fit_transform(wells['Formation/Reservoir'].values)

#st.write(wells)

# Splitting features and targets
df = wells.iloc[:, 1:]  # Ignoring lease name

target_feature = '# Clusters per Stage'
X = df.drop([target_feature], axis=1)
y = df[target_feature]


def scores(param_name, param_range):
    model = DecisionTreeRegressor(random_state=5)
    scorer = make_scorer(mean_squared_error)

    # LOOCV is CV with N folds. N number of data = 43
    train_scores, test_scores = validation_curve(model, X, y,
                                                 param_name=param_name,
                                                 param_range=param_range,
                                                 cv=53, scoring=scorer)

    train_score = np.mean(train_scores, axis=1)
    test_score = np.mean(test_scores, axis=1)
    return train_score, test_score


plt.rcParams['font.size'] = 20

# Range for parameters
param_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Scores for varying max_depth
train_score1, test_score1 = scores('max_depth', param_range)

# Scores for varying min_samples_leaf
train_score2, test_score2 = scores('min_samples_leaf', param_range)

# Plot validation curves
fig1 = plt.figure(figsize=(9, 4))
plt.subplot(1, 2, 1)
plt.plot(param_range, train_score1, label='train')
plt.plot(param_range, test_score1, label='test')
plt.ylim(0, 5.2)
plt.xlabel('max_depth')
plt.ylabel('MSE')
plt.legend(loc='lower right', fontsize=15)
plt.subplot(1, 2, 2)
plt.plot(param_range, train_score2, label='train')
plt.plot(param_range, test_score2, label='test')
plt.ylim(0, 5.2)
plt.xlabel('min_samples_leaf')
plt.legend(loc='lower right', fontsize=15)

st.header('Plot validation curves', divider='orange')
st.pyplot(fig1)


# BOSSIER_SHALE: 0, EAGLE FORD: 1, HAYNESVILLE SHALE: 2,
# MARCELLUS: 3, MARCELLUS_UPPER: 4

def predict(model, formation, temp, netpay, oilsat, gassat, sg,
            co2, n2, oilapi, tvd, lateral, topperf, botperf):
    X_test = np.array([formation, temp, netpay, oilsat, gassat, sg,
                       co2, n2, oilapi, tvd, lateral, topperf, botperf])
    X_test = X_test.reshape(1, -1)

    # model = DecisionTreeRegressor(max_depth=5, min_samples_leaf=4, random_state=5)
    # model.fit(X, y)
    y_pred = model.predict(X_test)
    return y_pred


# Predict on new input

# Define model
dt = DecisionTreeRegressor(max_depth=5, min_samples_leaf=4, random_state=5)

# Fit model to data
dt = dt.fit(X, y)

# New input

formation = st.number_input("Insert a formation")
temp = st.number_input("Insert a Temperature")
netpay = st.number_input("Insert a NetPay")
oilsat = st.number_input("Insert a Oil Saturation")
gassat = st.number_input("Insert a Gas Saturation")
sg = st.number_input("Insert a SG")
co2 = st.number_input("Insert a CO2")
n2 = st.number_input("Insert a N2")
oilapi = st.number_input("Insert a Oil API")
tvd = st.number_input("Insert a TVD")
lateral = st.number_input("Insert a lateral")
topperf = st.number_input("Insert a topperf")
botperf = st.number_input("Insert a botperf")

#st.write("The current number is ", number)

# formation = 3
# temp = 150
# netpay = 130
# oilsat = 0
# gassat = 0.6
# sg = 0.6
# co2 = 0.03
# n2 = 0.004
# oilapi = 30
# tvd = 8000
# lateral = 2000
# topperf = 6000
# botperf = 16000

h = predict(dt, formation, temp, netpay, oilsat, gassat, sg, co2, n2, oilapi,
            tvd, lateral, topperf, botperf)

st.header('Predict on max_depth', divider='orange')
st.write(h)
