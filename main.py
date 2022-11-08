import config
import pickle
from Project.utils import wine
from flask import Flask,request,jsonify,render_template
import numpy as np
log_model=pickle.load(open("logistic_model.pkl","rb"))

app=Flask(__name__)

@app.route('/')
def sms():
    return render_template('index.html')

@app.route('/output',methods=["POST"])
def result():
    x=[float(i) for i in request.form.values()]
    pred=log_model.predict([np.array(x)])
    return render_template('index.html',ans=pred)
if __name__=="__main__":
    app.run(port=config.PORT_NUMBER,host='0.0.0.0')
