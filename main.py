from flask import Flask,request,render_template,redirect,url_for
import pickle,gzip
import joblib
import numpy as np
model = joblib.load('logistic.pkl')

app=Flask(__name__)
@app.route('/')

def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
        rate_marriage=float(request.form.get('rate_marriage',False))
        age= float(request.form.get('age',False))
        yrs_married= float(request.form.get('yrs_married',False))
        children= float(request.form.get('children',False))
        religious=float(request.form.get('religious',False))
        educ= float(request.form.get('educ',False))
        occupation= float(request.form.get('occupation',False))
        occupation_husb= float(request.form.get('occupation_husb',False))

        arr=np.array([[rate_marriage,age,yrs_married,children,religious,educ,occupation,occupation_husb]])
        pred=model.predict(arr)
        if pred==0:
            res_Val="no affair"
        else:
            res_Val="affair"

        return render_template('index.html',prediction_text='Women have {}'.format(res_Val))


if __name__=='__main__':
    app.run(debug=True)

