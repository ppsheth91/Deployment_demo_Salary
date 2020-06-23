#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np

from flask import Flask, request, jsonify, render_template 
import pickle


# In[14]:


import os
os.chdir("E:/Data Science/End to End Deployment/ML Models/Salary Prediction_HR")


# In[15]:


app = Flask(__name__)  # Creating the flask web app #
model = pickle.load(open('Salary_model.pkl','rb'))


# In[16]:


@app.route('/')  # routing the command to the index / HTML page 
def home():
    return render_template('index.html')


# In[17]:


# Routing the intput features for this model #

@app.route('/predict',methods=['POST'])  # rputing the input features & hitting the predict funtion
def predict():  # predict function
    int_features = [int(x) for x in request.form.values()]  # taking tge features from input forms (web page) and converting into integers
    final_features = [np.array(int_features)]  # converting the inputs into an array (2D) shape for the model#
    prediction = model.predict(final_features) # Predicting the input fpor the features
    Output = round(prediction[0],2)  # converting the ioutput into 2 decimal point rounds
     
    return render_template('index.html',prediction_text= 'Employee salary should be $ {}'.format(Output)) # calling tyhe index template to display yhe prediced salary #


# In[26]:


if __name__ == "__main__":
    app.run(debug=True)


# In[ ]:




