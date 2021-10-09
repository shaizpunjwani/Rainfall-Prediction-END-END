from flask import Flask, render_template, request
import numpy as np
import pickle
import sklearn
from sklearn.preprocessing import MinMaxScaler

app=Flask(__name__)
model = pickle.load(open('rainfall_prediction.pkl', 'rb'))

#url
#app.route("/", methods=['GET','POST'])

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        feat1=int(request.form['feat1'])
        feat2=int(request.form['feat2'])
        feat3=int(request.form['feat3'])
        feat4=int(request.form['feat4'])
        feat5=int(request.form['feat5'])
        feat6=int(request.form['feat6'])
        feat7=int(request.form['feat7'])
        feat8=int(request.form['feat8'])
        feat9=int(request.form['feat9'])
        feature=np.array([[feat1],[feat2],[feat3],[feat4],[feat5],[feat6],[feat7],[feat8],[feat9]])
        scale=MinMaxScaler()
        feature_scale=scale.fit_transform(feature)
        feature_reshape=feature_scale.reshape(1,9)

        prediction=model.predict(feature_reshape)
        if prediction==0:
            return render_template('index.html',names='Tomorrow there will be no rain ')
        if prediction==1:
            return render_template('index.html',names='Tomorrow there will be rain. Get your UMBRELLAS OUT!!!!')
       
    return render_template('index.html')

#@app.route("/output.html")
# def output():
#     return render_template('output.houtput

if __name__=='__main__':
    #just to reload the server by itself when we made any changes
    app.run(debug=True)