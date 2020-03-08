import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

diabetes = pd.read_csv('diabetes.csv')

dataset_X = diabetes.iloc[:,[0, 1, 2, 3, 4, 5, 6, 7]].values

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
dataset_scaled = scaler.fit_transform(dataset_X)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict( scaler.transform(final_features) )

    if prediction == 1:
        pred = "You could have diabetes, please go and consult your GP."
    elif prediction == 0:
        pred = "You do not have Diabetes."
    output = pred
    return render_template('index.html',prob=positive_percent, prediction_text='{}'.format(output))

if __name__ == "__main__":
    app.run()
