from dis import show_code
import os
import shutil
import time
from flask_apscheduler import APScheduler
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import requests
from flask import Flask, render_template, request, redirect, flash, send_from_directory
from werkzeug.utils import secure_filename

from data import disease_map, details_map



# Database libraries
from flask import Flask, render_template, request, redirect, url_for, session
import pymysql as MySQLdb
import MySQLdb.cursors
import re
from flask_mysqldb import MySQL


# Download Model File
if not os.path.exists('model.h5'):
    print("Downloading model...")
    url = "https://drive.google.com/uc?id=1JNggWQ9OJFYnQpbsFXMrVu-E-sR3VnCu&confirm=t"
    r = requests.get(url, stream=True)
    with open('./model.h5', 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    print("Finished downloading model.")

# Load model from downloaded model file
model = load_model('model.h5')

# Create folder to save images temporarily
if not os.path.exists('./static/test'):
        os.makedirs('./static/test')

def predict(test_dir):
    test_img = [f for f in os.listdir(os.path.join(test_dir)) if not f.startswith(".")]
    test_df = pd.DataFrame({'Image': test_img})
    
    test_gen = ImageDataGenerator(rescale=1./255)

    test_generator = test_gen.flow_from_dataframe(
        test_df, 
        test_dir, 
        x_col = 'Image',
        y_col = None,
        class_mode = None,
        target_size = (256, 256),
        batch_size = 20,
        shuffle = False
    )
    predict = model.predict(test_generator, steps = np.ceil(test_generator.samples/20))
    test_df['Label'] = np.argmax(predict, axis = -1) # axis = -1 --> To compute the max element index within list of lists
    test_df['Label'] = test_df['Label'].replace(disease_map)

    prediction_dict = {}
    for value in test_df.to_dict('index').values():
        image_name = value['Image']
        image_prediction = value['Label']
        prediction_dict[image_name] = {}
        prediction_dict[image_name]['prediction'] = image_prediction
        prediction_dict[image_name]['description'] = details_map[image_prediction][0]
        prediction_dict[image_name]['symptoms'] = details_map[image_prediction][1]
        prediction_dict[image_name]['source'] = details_map[image_prediction][2]
    return prediction_dict


# Create an app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024 # maximum upload size is 50 MB
app.secret_key = "agentcrop"
ALLOWED_EXTENSIONS = {'png', 'jpeg', 'jpg'}
folder_num = 0
folders_list = []



#  ----------- DB configs -------------------

# Change this to your secret key (can be anything, it's for extra protection)
app.secret_key = 'your secret key'

# Enter your database connection details below
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'admin'
app.config['MYSQL_DB'] = 'cropdb'

# Intialize MySQL
mysql = MySQL(app)
# --------- FB Configs end -------------------

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# initialize scheduler
scheduler = APScheduler()
scheduler.api_enabled = True
scheduler.init_app(app)

# Adding Interval Job to delete folder
@scheduler.task('interval', id='clean', seconds=1800, misfire_grace_time=900)
def clean():
    global folders_list
    try:
        for folder in folders_list:
            if (time.time() - os.stat(folder).st_ctime) / 3600 > 1:
                shutil.rmtree(folder)
                folders_list.remove(folder)
                print("\n***************Removed Folder '{}'***************\n".format(folder))
    except:
        flash("Something Went Wrong! couldn't delete data!")

scheduler.start()
@app.route('/alldata' , methods=['GET', 'POST'])

def all_data():
        global folder_num
        global folders_list
        if request.method == 'POST':
            if folder_num >= 1000000:
                folder_num = 0
        app.config['UPLOAD_FOLDER'] = "./static/ok"
        app.config['UPLOAD_FOLDER'] = app.config['UPLOAD_FOLDER'] + '/predict_' + str(folder_num).rjust(6, "0")
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
            folders_list.append(app.config['UPLOAD_FOLDER'])
            folder_num += 1
        try:
            if len(os.listdir(app.config['UPLOAD_FOLDER'])) > 0:
                diseases = predict(app.config['UPLOAD_FOLDER'])
                return render_template('show_prediction.html',
                folder = app.config['UPLOAD_FOLDER'],
                predictions = diseases)
        except:
            return redirect('/')
        
        return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
# def home():
#     return render_template('index.html', msg='')

def get_disease():
    global folder_num
    global folders_list
    if request.method == 'POST':
        if folder_num >= 1000000:
            folder_num = 0
        # check if the post request has the file part
        if 'hiddenfiles' not in request.files:
            flash('No files part!')
            return redirect(request.url)
        # Create a new folder for every new file uploaded,
        # so that concurrency can be maintained
        # os.remove("./static/test/predict_")
        files = request.files.getlist('hiddenfiles')
        shutil.rmtree("./static/test")
        app.config['UPLOAD_FOLDER'] = "./static/test"

        app.config['UPLOAD_FOLDER'] = app.config['UPLOAD_FOLDER'] + '/predict_' + str(folder_num).rjust(6, "0")
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
            folders_list.append(app.config['UPLOAD_FOLDER'])
            folder_num += 1
        for file in files:
            if file.filename == '':
                flash('No Files are Selected!')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            else:
                flash("Invalid file type! Only PNG, JPEG/JPG files are supported.")
                return redirect('/')
        try:
            if len(os.listdir(app.config['UPLOAD_FOLDER'])) > 0:
                diseases = predict(app.config['UPLOAD_FOLDER'])
                return render_template('show_prediction.html',
                folder = app.config['UPLOAD_FOLDER'],
                predictions = diseases)
        except:
            return redirect('/')
        
    return render_template('index.html')

@app.route('/favicon.ico')

def favicon(): 
    return send_from_directory(os.path.join(app.root_path, 'static'), 'LeafX-Icon.png')

#API requests are handled here
@app.route('/api/predict', methods=['POST'])

def api_predict():  

    global folder_num
    global folders_list
    if folder_num >= 1000000:
            folder_num = 0
    # check if the post request has the file part
    if 'files' not in request.files:
        return {"Error": "No files part found."}
    # Create a new folder for every new file uploaded,
    # so that concurrency can be maintained
    files = request.files.getlist('files')
    app.config['UPLOAD_FOLDER'] = "./static/test"
    app.config['ALL_DATA'] = "./static/ok"
    app.config['UPLOAD_FOLDER'] = app.config['UPLOAD_FOLDER'] + '/predict_' + str(folder_num).rjust(6, "0")
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
        folders_list.append(app.config['UPLOAD_FOLDER'])
        folder_num += 1
    for file in files:
        if file.filename == '':
            return {"Error": "No Files are Selected!"}
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        else:
            return {"Error": "Invalid file type! Only PNG, JPEG/JPG files are supported."}
    try:
        if len(os.listdir(app.config['UPLOAD_FOLDER'])) > 0:
            diseases = predict(app.config['UPLOAD_FOLDER'])
            return diseases
    except:
        return {"Error": "Something Went Wrong!"}

@app.route('/home', methods=['GET', 'POST'])
def home():
    return render_template('index.html', msg='')

#  login page Route 

# http://localhost:5000/login/ - the following will be our login page, which will use both GET and POST requests
@app.route('/login/', methods=['GET', 'POST'])
def login():
    # Output message if something goes wrong...
    msg = ''

    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']

        # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = %s AND password = %s', (username, password,))
        # Fetch one record and return result
        account = cursor.fetchone()

                # If account exists in accounts table in out database
        if account:
            # Create session data, we can access this data in other routes
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            # Redirect to home page
            return redirect(url_for('home'))
        else:
            # Account doesnt exist or username/password incorrect
            msg = 'Incorrect username/password!'

    return render_template('login.html', msg='')

# Logout method 

# http://localhost:5000/python/logout - this will be the logout page
@app.route('/logout')
def logout():
    # Remove session data, this will log the user out
   session.pop('loggedin', None)
   session.pop('id', None)
   session.pop('username', None)
   # Redirect to login page
   return redirect(url_for('login'))


#  Register method

# http://localhost:5000/register - this will be the registration page, we need to use both GET and POST requests
@app.route('/register/', methods=['GET', 'POST'])
def register():
    # Output message if something goes wrong...
    msg = ''
    # Check if "username", "password" and "email" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']

        # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = %s', (username,))
        account = cursor.fetchone()
        # If account exists show error and validation checks
        if account:
            msg = 'Account already exists!'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers!'
        elif not username or not password or not email:
            msg = 'Please fill out the form!'
        else:
            # Account doesnt exists and the form data is valid, now insert new account into accounts table
            cursor.execute('INSERT INTO accounts VALUES (NULL, %s, %s, %s)', (username, password, email,))
            mysql.connection.commit()
            msg = 'You have successfully registered!'

    elif request.method == 'POST':
        # Form is empty... (no POST data)
        msg = 'Please fill out the form!'
    # show_code registration form with message (if any)
    return render_template('register.html', msg=msg)


#  Profile method

# http://localhost:5000//profile - this will be the profile page, only accessible for loggedin users
@app.route('/profile')
def profile():
    # Check if user is loggedin
    if 'loggedin' in session:
        # We need all the account info for the user so we can display it on the profile page
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE id = %s', (session['id'],))
        account = cursor.fetchone()
        # Show the profile page with account info
        return render_template('profile.html', account=account)
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))