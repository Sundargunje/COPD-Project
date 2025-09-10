
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__) 
model = pickle.load(open('backend/copd_disease_model.pkl', 'rb'))

@app.route('/')
def home():
    precision = 0.45
    recall = 0.42
    f1_score = 0.42
    return render_template('home.html', precision=precision, recall=recall, f1_score=f1_score)

@app.route('/predict', methods=['POST'])
def predict():
    d1 = 1 if request.form['smoke'] == "Yes" else 0
    d2 = float(request.form['fvc'])
    d3 = float(request.form['fec1'])
    d4 = 1 if request.form['pefr'] == "Yes" else 0
    d5 = 1 if request.form['o2'] == "Yes" else 0
    d6 = 1 if request.form['abgO2'] == "Yes" else 0
    d7 = 1 if request.form['abgCO2'] == "Yes" else 0
    d8 = 1 if request.form['abgPH'] == "Acidic" else 0
    d9 = 1 if request.form['asthama'] == "Yes" else 0
    d10 = 1 if request.form['otherDiseases'] == "Yes" else 0
    d11 = float(request.form['age'])

    # Create an array for the model
    arr = np.array([[d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11]])

    # Make a prediction
    pred = model.predict(arr)

    # Render the result on the page
    return render_template('after.html', data=pred[0])

if __name__ == "__main__":
    app.run(debug=True)