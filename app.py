import pickle
from flask import Flask, request, app, render_template, redirect, url_for
import numpy as np
import pandas as pd
import gspread
from sklearn.impute import SimpleImputer

app = Flask(__name__)

# Load the model
modeljee = pickle.load(open("modeljee.pkl", "rb"))
modelapeamcet = pickle.load(open("modelapeamcet.pkl", "rb"))
modelsrm = pickle.load(open("modelsrm.pkl", "rb"))
modelvit = pickle.load(open("modelvit.pkl", "rb"))

def assign_college_branch(score):
    if score >= 400:
        return 'BITS Pilani', 'CSE'
    elif 350 <= score < 400:
        return 'BITS Pilani', 'ECE'
    elif 300 <= score < 350:
        return 'BITS Goa', 'CSE'
    elif 250 <= score < 300:
        return 'BITS Goa', 'ECE'
    elif 200 <= score < 250:
        return 'BITS Hyderabad', 'Mechanical'
    else:
        return 'BITS Hyderabad', 'Chemical'

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/jee')
def jee():
    return render_template('jee.html')
@app.route('/ap')
def ap():
    return render_template('apeamcet.html')   
@app.route('/bits')
def bits():
    return render_template('bits.html')   
@app.route('/srm')
def srm():
    return render_template('srm.html')   
@app.route('/vit')
def vit():
    return render_template('vit.html')
@app.route('/ts')
def ts():
    return render_template('tseamcet.html') 

@app.route('/loginres')
def loginres():
    name = request.form.get('username')
    password = request.form.get('password')
    return redirect(url_for('home'))

@app.route('/predictjee', methods = ['POST'])
def predictjee():

    Category = {'0':'General', '1':'Other Backward Classes-Non Creamy Layer', '6':'Scheduled Castes', '8':'Scheduled Tribes',
                '3':'General & Persons with Disabilities', '5':'Other Backward Classes & Persons with Disabilities', 
                '7':'Scheduled Castes & Persons with Disabilities', '9':'Scheduled Tribes & Persons with Disabilities',
                '1':'General & Economically Weaker Section', '2':'General & Economically Weaker Section & Persons with Disability'}
    
    Quota = {'0':'All-India', '3':'Home-State', '1':'Andhra Pradesh', '2':'Goa', '4':'Jammu & Kashmir', '5':'Ladakh'}

    Pool = {'0':'Neutral', '1':'Female Only'}

    Institute = {'0':'IIT', '1':'NIT'}

    data = [x for x in request.form.values()]
    list1 = data.copy()

    list1[2] = Category.get(list1[2])
    list1[3] = Quota.get(list1[3])
    list1[4] = Pool.get(list1[4])
    list1[5] = Institute.get(list1[5])

    data1 = [float(x) for x in data]

    final_output = np.array(data1).reshape(1, -1)
    output = modeljee.predict(final_output)[0]
    
    list1.append(output[0])
    list1.append(output[1])
    list1.append(output[2])
    return render_template("prediction.html", prediction_college = "College : {}".format(output[0]) ,prediction_branch="  Degree : {}".format(output[1]) ,predicted_Course="Course : {}".format(output[2]), prediction = "Thank you, Hope this will match your requirement !!!")

@app.route('/predictap', methods=['POST'])
def predictap():
    Category = {
        '0': 'General', '0': 'SC', '1': 'ST', '2': 'pwd', '3': 'Economically weaker',
        '4': 'BC-A', '5': 'BC-B', '6': 'BC-C', '7': 'BC-D', '8': 'BC-E'
    }
    Gender = {'0':'BOYS','1':'GIRLS'}

    # Get form data
    ap_rank = request.form.get('rank')
    gender = request.form.get('gender')
    category =request.form.get('category')
    # gender1=Gender.get(gender)
    # category1=Category.get(category)
    ap_rank1=int(ap_rank)
    gender2=int(gender)
    category2=int(category)

    # Validate input data
    if gender is None or category is None or ap_rank is None:
        return "Error: Incomplete form submission"

    # Construct input data as numpy array
    input_data = np.array([[category2, gender2, ap_rank1]])

    # Assuming modelapeamcet.predict returns a list or tuple
    output = modelapeamcet.predict(input_data)

    

    prediction_college = "College : {}".format(output[0][0])
    predicted_course = "Course : {}".format(output[0][1])

    return render_template("prediction.html",
                           prediction_college=prediction_college,
                           predicted_Course=predicted_course,
                           prediction="Thank you, hope this matches your requirements!")

@app.route('/predictbits', methods=['POST'])
def predictbits():
    
    bits_score = request.form.get('rank',type=int)


    output=assign_college_branch(bits_score)

    prediction_college = "College : {}".format(output[0])
    predicted_course = "Course : {}".format(output[1])

    return render_template("prediction.html",
                            prediction_college=prediction_college,
                            predicted_Course=predicted_course,
                            prediction="Thank you, hope this matches your requirements!")
@app.route('/predictsrm', methods=['POST'])
def predictsrm():
    Category = {
        '0': 'General', '0': 'SC', '1': 'ST', '2': 'pwd', '3': 'Economically weaker',
        '4': 'BC-A', '5': 'BC-B', '6': 'BC-C', '7': 'BC-D', '8': 'BC-E'
    }
    Gender = {'0':'BOYS','1':'GIRLS'}

    # Get form data
    srm_rank = request.form.get('rank')
    gender = request.form.get('gender')
    category =request.form.get('category')
    # gender1=Gender.get(gender)
    # category1=Category.get(category)
    srm_rank1=int(srm_rank)
    gender2=int(gender)
    category2=int(category)

    # Validate input data
    if gender is None or category is None or srm_rank is None:
        return "Error: Incomplete form submission"

    # Construct input data as numpy array
    input_data = np.array([[category2, gender2, srm_rank1]])

    # Assuming modelapeamcet.predict returns a list or tuple
    output = modelsrm.predict(input_data)

    

    prediction_college = "College : {}".format(output[0][0])
    predicted_course = "Course : {}".format(output[0][1])

    return render_template("prediction.html",
                           prediction_college=prediction_college,
                           predicted_Course=predicted_course,
                           prediction="Thank you, hope this matches your requirements!")

@app.route('/predictvit', methods=['POST'])
def predictvit():
    Category = {
        '0': 'General', '0': 'SC', '1': 'ST', '2': 'pwd', '3': 'Economically weaker',
        '4': 'BC-A', '5': 'BC-B', '6': 'BC-C', '7': 'BC-D', '8': 'BC-E'
    }
    Gender = {'0':'BOYS','1':'GIRLS'}

    # Get form data
    vit_rank = request.form.get('rank')
    gender = request.form.get('gender')
    category =request.form.get('category')
    # gender1=Gender.get(gender)
    # category1=Category.get(category)
    vit_rank1=int(vit_rank)
    gender2=int(gender)
    category2=int(category)

    # Validate input data
    if gender is None or category is None or vit_rank is None:
        return "Error: Incomplete form submission"

    # Construct input data as numpy array
    input_data = np.array([[category2, gender2, vit_rank1]])

    # Assuming modelapeamcet.predict returns a list or tuple
    output = modelvit.predict(input_data)

    

    prediction_college = "College : {}".format(output[0][0])
    predicted_course = "Course : {}".format(output[0][1])

    return render_template("prediction.html",
                           prediction_college=prediction_college,
                           predicted_Course=predicted_course,
                           prediction="Thank you, hope this matches your requirements!")


if __name__ == '__main__':
    app.run(debug = True)