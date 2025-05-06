from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

ridge_model = pickle.load(open('Models/ridge.pkl',"rb"))
standard_scaler = pickle.load(open('Models/scaler.pkl',"rb"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods = ["GET","POST"])
def predict_datapoint():
    if request.method == "POST":
        Tempurature = float(request.form.get('Tempurature'))
        RH = float(request.form.get('RH'))
        WH = float(request.form.get('WH'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data_scaled =  standard_scaler.transform([[Tempurature,RH,WH,Rain,FFMC,DMC,ISI,Classes,Region]])
        result = ridge_model.predict(new_data_scaled)

        return render_template('home.html',result = result[0])

    else:
        return render_template("home.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)
