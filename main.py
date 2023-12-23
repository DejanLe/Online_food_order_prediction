# https://thecleverprogrammer.com/2022/06/07/online-food-order-prediction-with-machine-learning/
import pandas as pd 
import numpy as np 
from flask import Flask, request, render_template

import requests

import pandas as pd

import pickle
'''
print("Enter Customer Details to Predict If the Customer Will Order Again")
a = int(input("Enter the Age of the Customer: "))
b = int(input("Enter the Gender of the Customer (1 = Male, 0 = Female): "))
c = int(input("Marital Status of the Customer (1 = Single, 2 = Married, 3 = Not Revealed): "))
d = int(input("Occupation of the Customer (Student = 1, Employee = 2, Self Employeed = 3, House wife = 4): "))
e = int(input("Monthly Income: "))
f = int(input("Educational Qualification (Graduate = 1, Post Graduate = 2, Ph.D = 3, School = 4, Uneducated = 5): "))
g = int(input("Family Size: "))
h = int(input("Pin Code: "))
i = int(input("Review of the Last Order (1 = Positive, 0 = Negative): "))
'''
app = Flask(__name__)
data = pd.read_csv('onlinefoodscsv.csv').dropna().drop_duplicates()


data["Gender"] = data["Gender"].map({"Male": 1, "Female": 0})
data["Marital Status"] = data["Marital Status"].map({"Married": 2, 
                                                     "Single": 1, 
                                                     "Prefer not to say": 0})
data["Occupation"] = data["Occupation"].map({"Student": 1, 
                                             "Employee": 2, 
                                             "Self Employeed": 3, 
                                             "House wife": 4})
data["Educational Qualifications"] = data["Educational Qualifications"].map({"Graduate": 1, 
                                                                             "Post Graduate": 2, 
                                                                             "Ph.D": 3, "School": 4, 
                                                                             "Uneducated": 5})
data["Monthly Income"] = data["Monthly Income"].map({"No Income": 0, 
                                                     "25001 to 50000": 5000, 
                                                     "More than 50000": 7000, 
                                                     "10001 to 25000": 25000, 
                                                     "Below Rs.10000": 10000})
data["Feedback"] = data["Feedback"].map({"Positive": 1, "Negative ": 0})
#print(data.head())

@app.route('/')

def index(): 
    return render_template('index.html')


@app.route('/predictions',  methods=['POST','GET'])
 
def recommend():
    a = request.form['a']
    b = request.form['b']
    c = request.form['c']
    d = request.form['d']
    e = request.form['e']
    f = request.form['f']
    g = request.form['g']
    h = request.form['h']
    i = request.form['i']



    from sklearn.model_selection import train_test_split
    x = np.array(data[["Age", "Gender", "Marital Status", "Occupation", 
                   "Monthly Income", "Educational Qualifications", 
                   "Family size", "Pin code", "Feedback"]])
    y = np.array(data[["Output"]])
    #train machine learning model

    from sklearn.ensemble import RandomForestClassifier 
    xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.10, random_state=42)

    model = RandomForestClassifier()
    model.fit(xtrain, ytrain)
    print(model.score(xtest, ytest))
    '''
    print("Enter Customer Details to Predict If the Customer Will Order Again")
    a = int(input("Enter the Age of the Customer: "))
    b = int(input("Enter the Gender of the Customer (1 = Male, 0 = Female): "))
    c = int(input("Marital Status of the Customer (1 = Single, 2 = Married, 3 = Not Revealed): "))
    d = int(input("Occupation of the Customer (Student = 1, Employee = 2, Self Employeed = 3, House wife = 4): "))
    e = int(input("Monthly Income: "))
    f = int(input("Educational Qualification (Graduate = 1, Post Graduate = 2, Ph.D = 3, School = 4, Uneducated = 5): "))
    g = int(input("Family Size: "))
    h = int(input("Pin Code: "))
    i = int(input("Review of the Last Order (1 = Positive, 0 = Negative): "))
    '''
    features = np.array([[a, b, c, d, e, f, g, h, i]])
    print("Finding if the customer will order again: ", model.predict(features))
    prediction = model.predict(features)
    return  render_template('predictions.html', prediction=prediction)



if __name__ == '__main__':
    app.run(debug=True)









