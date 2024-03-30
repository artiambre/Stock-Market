import Model_building as mb
from flask import Flask, render_template, request, redirect, url_for, flash
from flask.globals import session
import pickle , re, os
import pandas as pd
from csv import reader
import flask
import numpy as np
import os.path
import Prediction
from datetime import datetime
import sklearn
import json
import flask.views
from joblib import dump, load
from flask import jsonify
import math

app = Flask(__name__,template_folder='template')
app.secret_key = '1F4453C6EA2C5B454D221285FFFFC'

@app.route('/')  
def index():
    if 'username' in session and session['username'] != 'admin':
        return redirect(url_for('user'))
    elif 'username' in session and session['username'] == 'admin':
        return redirect(url_for('admin'))
    else:
        return render_template('login.html')
    
@app.route('/login_nav', methods=['GET','POST'])
def login_nav(): 
    msg='Fill details..'
    return render_template('login.html', msg = msg)

@app.route('/login', methods=['GET','POST'])
def login(): 
    username=request.form['username']
    password = request.form['password']
    if username=="admin" and password=="admin":
        session['username'] = username
        return redirect(url_for('admin'))
    
    else:
            msg='login Failed'
            return redirect(url_for('login_nav'))

@app.route('/_compute')
def prediction():
    sentence = flask.request.args.get('sentence')

    status,probability=detecting_fake_news(sentence)
    n=0
    nn=probability*100
    result=float("{:.2f}".format(nn))
    print(status)
    print(nn)
    if status==True:
        print("True news detected.")
        n=100
    else :
        print("Alert!..Fake news detected.")
        n=-100
    print(n)
    return jsonify(result=result)


#function to run for prediction
def detecting_fake_news(var):    
#retrieving the best model for prediction call
    load_model = pickle.load(open('final_model.sav', 'rb'))
    prediction = load_model.predict([var])
    prob = load_model.predict_proba([var])
    return prediction[0],prob[0][1]

@app.route('/admin')
def admin():
    if 'username' in session:
        user = session['username']
        return render_template('admin_home.html')
    else:
        return redirect(url_for('login_nav'))

@app.route('/news_nav')
def news_nav():
    if 'username' in session:
        user = session['username']
        return render_template('index.html')
    else:
        return redirect(url_for('login_nav'))

@app.route('/train_model')
def train_model():
    myarray=[]
    if 'username' in session:
        return render_template('stock_analysis.html', DataOut= myarray)
    else:
        return redirect(url_for('login_nav'))

@app.route('/train_model_save',methods=["GET","POST"])
def train_model_save():
    
    company=request.form['stock']
    index_=request.form['index']
    comp_name  = company.capitalize()
    if index_ == "bse":
        comp_name =company.capitalize()+".BS"
    elif index_ == "nse":
        comp_name = company.capitalize() + ".NS"
    file_exist_status=os.path.exists(comp_name+".h5")
    if file_exist_status==True:
        return render_template('stock_analysis.html', DataOut= "Model already trained...")
    else:
        result=mb.model_building(company,index_)
        if 'username' in session:
            return render_template('stock_analysis.html', DataOut= result)
        else:
            return redirect(url_for('login_nav'))


@app.route('/stockPredict_nav')
def stockPredict_nav():
        return render_template('stockpredict.html', DataOut= "")

@app.route('/predictStock',methods=["GET","POST"])
def predictStock():

    company=request.form['company']
    market=request.form['index']
    date_=request.form['date']
    comp_name  = company.capitalize()
    if market == "bse":
        comp_name =company.capitalize()+".BS"
    elif market == "nse":
        comp_name = company.capitalize() + ".NS"
    print(comp_name,date_,market)
    file_exist_status=os.path.exists(comp_name+".h5")
    if file_exist_status==False:
        return render_template('stockpredict.html', DataOut= "Please train model for "+company)
    else:
        if 'username' in session:
            updated_date = datetime.strptime(date_, "%Y-%m-%d").strftime('%d%m%Y')
            predictions,actual=Prediction.predict_stock(updated_date,comp_name)
            print(predictions[0][0],actual[0])
            data={}
            if isinstance(actual[0], float):

                data={"Date":date_,"Predicted Value":predictions[0][0],"Actual Value":actual[0]}
            else:
                data={"Date":date_,"Predicted Value":predictions[0][0],"Actual Value": "Can't Fetch Data"}
            return render_template('stockpredict.html', DataOut= data)
        else:
            return redirect(url_for('login_nav'))



@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login_nav')) 

        
if __name__ == '__main__':  
    app.run(debug=True)
