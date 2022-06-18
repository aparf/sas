from flask import Flask, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
#You need to use following line [app Flask(__name__]

UPLOAD_FOLDER = './static/uploads'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

import os
import numpy as np
import pickle
import sklearn 
import xgboost as xgb
import pandas as pd
from io import StringIO
from sklearn.preprocessing import LabelEncoder


# prediction function
def ValuePredictor(to_predict_list, modelType):
    # to_predict = np.array(to_predict_list).reshape(1, 19)
    # with open('model.json', 'rb') as f:
    loaded_model= xgb.Booster()
    if modelType == 1:
        loaded_model.load_model("model_1.bin")
    elif modelType == 2:
        loaded_model.load_model("model_2.bin")
    result = loaded_model.predict(xgb.DMatrix(to_predict_list))
    return result

def requestHandler(request):
    to_predict_list = request.form.to_dict()
    to_predict_list = list(to_predict_list.values())
    del to_predict_list[-1]
    return(list(map(float, to_predict_list)))

def requestHandlerFile(request):
    f = request.files['file']
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
    data = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], f.filename), delimiter=';')
    le = LabelEncoder()
    data[data.select_dtypes(include=object).columns] = data[data.select_dtypes(include=object).columns].apply(lambda col: le.fit_transform(col))
    return data



@app.route("/")
def index():
    return render_template('index.html')


@app.route('/result', methods = ['POST'])
def result():
    if request.method == 'POST':
        # to_predict_list = requestHandler(request)
        data = requestHandlerFile(request)
    
        

        # return 'file uploaded successfully'
        # # Handler for the first model:
        if request.form['model'] == '0':
            result = ValuePredictor(data, modelType=1)   
            name = 'model_1_results.csv' 
        # Handler for the second model:
        if request.form['model'] == '1':
            result = ValuePredictor(data, modelType=2)      
            name = 'model_2_results.csv' 
        # # if int(result) > 0.5:
        # #     text = "As the probability exceeds the threshold of 50%, the clients is classified as 1, namely we do expect the client to respond."
        # # else:
        # #     text = "As the probability is less than the threshold of 50%, the clients is classified as 0, namely we do not expect the client to respond."
        result = [(lambda i: 0 if i < 0.5 else i)(i) for i in result]
        result = [(lambda i: 1 if i >= 0.5 else i)(i) for i in result]
        np.savetxt("./static/uploads/" + name, result, delimiter=",")
        return send_file("./static/uploads/" + name, as_attachment=True)


if __name__ == '__main__':
   app.run(debug = True)



