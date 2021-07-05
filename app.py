from flask import Flask, render_template, request
import numpy as np
import joblib
import pickle 
import pandas as pd 


app = Flask(__name__)

model = joblib.load(open('marketing_model.pkl', 'rb'))


@app.route('/')
def man():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    global df

    input_features = [float(x) for x in request.form.values()]
    features_value = np.array(input_features)
    
    #validate input hours
    #if input_features[0] <0 or input_features[0] >24:
     #   return render_template('index.html', prediction_text='Please enter valid hours between 1 to 24 if you live on the Earth')
        

    output = model.predict([features_value])

    # input and predicted value store in df then save in csv file
    
    output = np.around([output], decimals=2) 


    return render_template('index.html', prediction_text='Your Sale is: {} '.format(output) )  








if __name__ == "__main__":
    app.run(debug=True)



