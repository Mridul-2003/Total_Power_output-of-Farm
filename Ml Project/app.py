from flask import Flask,render_template,request
import pickle
import numpy as np

app=Flask(__name__)

total_power_predict=pickle.load(open('Adelaide.pkl','rb'))

@app.route('/')
def main():
    return render_template("home.html")

#@app.route('/Predict/') 
@app.route('/predict/')
def predict():
    return render_template("Predict.html")

@app.route('/home/')
def home():
    return render_template("home.html")

@app.route('/about/')
def about():
    return render_template("about.html")

@app.route('/predictpower/',methods=['POST'])

def predictpower():
    int_features=[x for x in request.form.values()]
    processed_feature = np.array(int_features, dtype=float)
    if processed_feature.shape != (48,):
        processed_feature = processed_feature.reshape(48,)  # Reshape to (48,) array
    prediction = total_power_predict.predict(processed_feature.reshape(1, -1))
    output = round(prediction.item(), 2)
    return render_template("Predict.html",output="{}".format(output))


if __name__=="__main__":
    app.run(debug=False,host='0.0.0.0')
