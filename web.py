from flask import Flask, jsonify, request
import traceback
import numpy as np
import joblib

import pandas as pd


app = Flask(__name__)

@app.route("/")
def hello():
    return "<h>API Homepage</h>"

@app.route("/predict", methods=['POST'])   #API endpoint will consist /predict
def predict():
    if lr:
        try:
            json_ = request.json
            print(json_)
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)
            prediction = list(lr.predict(query))
            return jsonify({'Prediction': str(prediction)})
        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        print('Train the model first')
        return('No model here to use')
    
    
if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except:
        port = 12345
    # deserialize model and columns
    lr = joblib.load('model.pkl')
    print('Model loaded')
    model_columns = joblib.load('model_columns.pkl')
    print('Model columns loaded')
    app.run(port=port,debug=True)
        