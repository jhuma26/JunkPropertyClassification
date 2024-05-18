import os
import sys
from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import json
import pickle
from pprint import pprint
import json
from sklearn.preprocessing import StandardScaler
from src.pipelines.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        req_data = json.loads(request.data.decode('utf-8'))
        req_data = req_data.get('data')
        data=CustomData(
            InteriorsStyle = req_data.get('InteriorsStyle'),
            PriceIndex8 = req_data.get('PriceIndex8'),
            ListDate = req_data.get('ListDate'),
            Material = req_data.get('Material'),
            PriceIndex9 = req_data.get('PriceIndex9'),
            Agency = req_data.get('Agency'),
            AreaIncomeType = req_data.get('AreaIncomeType'),
            EnvRating = req_data.get('EnvRating'),
            PriceIndex7 = req_data.get('PriceIndex7'),
            ExpeditedListing = req_data.get('ExpeditedListing'),
            PriceIndex4 = req_data.get('PriceIndex4'),
            PriceIndex1 = req_data.get('PriceIndex1'),
            PriceIndex6 = req_data.get('PriceIndex6'),
            PRIMEUNIT = req_data.get('PRIMEUNIT'),
            Channel = req_data.get('Channel'),
            Zip = req_data.get('Zip'),
            InsurancePremiumIndex = req_data.get('InsurancePremiumIndex'),
            PlotType = req_data.get('PlotType'),
            Architecture = req_data.get('Architecture'),
            PriceIndex3 = req_data.get('PriceIndex3'),
            Region = req_data.get('Region'),
            PriceIndex5 = req_data.get('PriceIndex5'),
            SubModel = req_data.get('SubModel'),
            Facade = req_data.get('Facade'),
            State = req_data.get('State'),
            NormalisedPopulation = req_data.get('NormalisedPopulation'),
            BuildYear = req_data.get('BuildYear'),
            RegionType = req_data.get('RegionType'),
            PropertyAge = req_data.get('PropertyAge'),
            PriceIndex2 = req_data.get('PriceIndex2')
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        print(results[0])
        return render_template('home.html',results=results[0])
        
    

if __name__=="__main__":
    app.run(host="0.0.0.0")      