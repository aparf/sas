from flask import Flask, render_template, request
#You need to use following line [app Flask(__name__]
 
app = Flask(__name__)

import numpy as np
import pickle
import sklearn 
import xgboost as xgb
# prediction function
def ValuePredictor(to_predict_list, modelType):
    to_predict = np.array(to_predict_list).reshape(1, 19)
    # with open('model.json', 'rb') as f:
    loaded_model= xgb.Booster()
    if modelType == 1:
        loaded_model.load_model("model_1.bin")
    elif modelType == 2:
        loaded_model.load_model("model_2.bin")
    result = loaded_model.predict(xgb.DMatrix(to_predict))
    return result[0]

def requestHandler(request):
    to_predict_list = request.form.to_dict()
    to_predict_list = list(to_predict_list.values())
    del to_predict_list[-1]
    return(list(map(float, to_predict_list)))


@app.route("/")
def index():
    return render_template('index.html')


@app.route('/result', methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = requestHandler(request)
        # Handler for the first model:
        if request.form['action'] == 'Model1':
            result = ValuePredictor(to_predict_list, modelType=1)       
            # if int(result)== 1:
            #     prediction = int(result)
            # else:
            #     prediction = int(result)           
        # Handler for the second model:
        if request.form['action'] == 'Model2':
            result = ValuePredictor(to_predict_list, modelType=2)       
        if int(result) > 0.5:
            text = "As the probability exceeds the threshold of 50%, the clients is classified as 1, namely we do expect the client to respond."
        else:
            text = "As the probability is less than the threshold of 50%, the clients is classified as 0, namely we do not expect the client to respond."

        return render_template("result.html", prediction = result, text = text)




