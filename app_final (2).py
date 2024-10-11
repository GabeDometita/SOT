from flask import Flask, request, render_template, send_from_directory, session, request, redirect, url_for, flash
from flask_session import Session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
import joblib
import os
import io
import base64
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pytz
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from datetime import datetime
from flask import jsonify
from flask_migrate import Migrate


matplotlib.use('Agg')


# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'your_secret_key'  

# Set up server-side session using Flask-Session
app.config['SESSION_TYPE'] = 'filesystem'  # Store session data on the server
app.config['SESSION_FILE_DIR'] = os.path.join(app.root_path, 'flask_session')  # Directory for session files
app.config['SESSION_PERMANENT'] = False  # Make sessions temporary
app.config['SESSION_USE_SIGNER'] = True  # Sign the session ID to prevent tampering

# Secret key
app.secret_key = 'your_secret_key'

# Configure SQLite Database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

Session(app)

db = SQLAlchemy(app)
migrate = Migrate(app, db)

class UserMetrics(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    feature_importance_plot = db.Column(db.Text, nullable=False)  # Base64-encoded image
    correlation_heatmap_plot = db.Column(db.Text, nullable=False)  # Base64-encoded image
    retention_by_course_plot = db.Column(db.Text, nullable=False)
    accuracy_by_course_plot = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship with the User model
    user = db.relationship('User', backref=db.backref('metricspage', lazy=True))

class UserResults(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    result_file = db.Column(db.String(255), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship with the User model
    user = db.relationship('User', backref=db.backref('results', lazy=True))
    
class UploadHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    file_name = db.Column(db.String(255), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship with the User
    user = db.relationship('User', backref=db.backref('uploads', lazy=True))    

class DownloadHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    file_name = db.Column(db.String(255), nullable=False)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(pytz.UTC))

    # Relationship with the User
    user = db.relationship('User', backref=db.backref('downloads', lazy=True))

# Define the User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(50), nullable=False, default='user')  # Add the role column

# ActivityLog model for tracking user activities
class ActivityLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    activity = db.Column(db.String(255), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship with the User
    user = db.relationship('User', backref=db.backref('activities', lazy=True))

# Method to hash the password
def set_password(self, password):
    self.password = generate_password_hash(password)

# Method to check the password
def check_password(self, password):
    return check_password_hash(self.password, password)

# Create the database tables
with app.app_context():
    db.create_all()


# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)

# Paths for model and metadata files
MODEL_FILE = 'models/pretrained_model_xgboost.pkl'
FEATURE_NAMES_FILE = 'models/feature_names.pkl'
IMPUTER_FILE = 'models/imputer.pkl'
TRAINING_GRAPHS_DIR = 'models/'  # Directory where pre-trained graphs are saved

# Define the color mapping for each course
course_colors = {
    "Architecture": "#FF6F61",
    "BSAeronautical": "#6B5B95",
    "CE": "#88B04B",
    "Comp. Eng'g.": "#FFA500",
    "EE": "#92A8D1",
    "ELECENG": "#034F84",
    "IE": "#F7786B",
    "ME": "#C94C4C"
}

@app.route('/downloads/<filename>')
def download_file(filename):
    # Log the download activity
    if 'user_id' in session:
        activity = ActivityLog(user_id=session['user_id'], activity=f"downloaded file: {filename}", timestamp=datetime.utcnow())
        db.session.add(activity)
        db.session.commit()

    return send_from_directory('downloads', filename)

@app.route('/', methods=['GET', 'POST'])
def homepage():
    if 'email' not in session:
        flash('Please log in to access this page.', 'warning')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        # Handle file upload
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Log the upload activity
        if 'user_id' in session:
            activity = ActivityLog(user_id=session['user_id'], activity=f"uploaded file: {file.filename}", timestamp=datetime.utcnow())
            db.session.add(activity)
            db.session.commit()

        # Load the test dataset
        test_data = pd.read_excel(filepath)

        # Load the test dataset (read the sheet names to infer the school year)
        xl = pd.ExcelFile(filepath)
        sheet_names = xl.sheet_names  # Get the sheet names from the Excel file

        # Assuming the sheet names are like '2324_1STSEM' or '2324_2NDSEM', extract the school year
        if sheet_names:
            # Extract the first four digits, convert to 20xx/20xx format
            start_year = int(sheet_names[0][:2]) + 2000
            end_year = start_year + 1
            school_year = f"{start_year}-{end_year}"
        else:
            school_year = "Unknown"

        institute = "SEA School"

        # Load the test data from the first sheet (assuming it’s the first sheet)
        test_data = xl.parse(sheet_names[0])

        # Total number of students in the dataset
        total_students = len(test_data)

        # Total number of unique courses in the dataset
        total_courses = len(test_data['COURSE'].unique()) if 'COURSE' in test_data.columns else 0

        # Load the test dataset
        test_data = pd.read_excel(filepath)

        # Load the trained model and necessary components
        model = joblib.load(MODEL_FILE)
        feature_names = joblib.load(FEATURE_NAMES_FILE)
        imputer = joblib.load(IMPUTER_FILE)

        # Preprocess test data
        test_data['GPA'] = pd.to_numeric(test_data['GPA'], errors='coerce')

        # One-hot encode categorical variables (Include GENDER and ENROLLMENT STATUS)
        test_data['GENDER'] = test_data['GENDER'].astype('category')
        test_data['ENROLLMENT STATUS'] = test_data['ENROLLMENT STATUS'].astype('category')
        test_data = pd.get_dummies(test_data, columns=['GENDER', 'ENROLLMENT STATUS'], drop_first=True)

        # Ensure all one-hot encoded columns are present, even if not in the test data
        for col in feature_names:
            if col not in test_data.columns:
                test_data[col] = 0  # Add missing columns with a default value of 0

        # Reorder columns to match the training data feature order
        X_test = test_data[feature_names]

        # Apply the imputer (same as in training)
        X_test = imputer.transform(X_test)

        # Make predictions on the test data
        y_test_pred = model.predict(X_test)
        retention_prediction = y_test_pred[0] if len(y_test_pred) == 1 else None

        # Calculate predicted overall retention rate (based on model predictions)
        predicted_retained_students = sum(y_test_pred)  # Predicted retained students
        total_students = len(y_test_pred)
        predicted_retention_rate = (predicted_retained_students / total_students) * 100

        # Calculate the real overall retention rate (if available)
        if 'RETAINED' in test_data.columns:
            real_retained_students = sum(test_data['RETAINED'])  # Real retained students from the actual data
            real_retention_rate = (real_retained_students / len(test_data)) * 100
        else:
            real_retention_rate = None

        # Calculate the prediction error
        prediction_error = predicted_retention_rate - real_retention_rate if real_retention_rate is not None else None

        # Retention prediction summary by course with distribution calculation
        retention_by_course_summary = []
        total_predicted_retained = sum(y_test_pred)
        if 'COURSE' in test_data.columns:
            courses = test_data['COURSE'].unique()

            for course in courses:
                course_data = test_data[test_data['COURSE'] == course]
                X_course = imputer.transform(course_data[feature_names])
                y_pred_course = model.predict(X_course)
                retained_count = sum(y_pred_course)
                real_retained_count = sum(course_data['RETAINED']) if 'RETAINED' in course_data.columns else None
                retention_distribution = (retained_count / total_predicted_retained) * 100 if total_predicted_retained > 0 else None

                retention_by_course_summary.append({
                    'course': course,
                    'predicted_retained_students': retained_count,
                    'total_students': len(course_data),
                    'real_retained_students': real_retained_count,
                    'retention_distribution': retention_distribution
                })

        # Generate plots and save to session
        # Feature Importance Plot
        feature_labels = [col.replace('GENDER_1', 'GENDER').replace('ENROLLMENT STATUS_1', 'ENROLLMENT STATUS') for col in feature_names]

        # Feature Importance Plot
        plt.figure(figsize=(8, 8))
        feature_importance = model.feature_importances_
        plt.pie(
            feature_importance,
            labels=feature_labels,  # Use modified labels for display
            autopct='%1.1f%%',
            startangle=90,
            counterclock=False,
            wedgeprops={'edgecolor': 'black'},
            colors=['#FF6F61', '#6B5B95', '#88B04B', '#F7CAC9', '#92A8D1', '#034F84', '#F7786B', '#C94C4C']
        )
        img_feature_importance = io.BytesIO()
        plt.savefig(img_feature_importance, format='png')
        img_feature_importance.seek(0)
        feature_importance_plot = base64.b64encode(img_feature_importance.getvalue()).decode('utf8')
        plt.close()

        # Correlation Heatmap
        numeric_data = test_data.select_dtypes(include=[np.float64, np.int64, np.uint8])
        numeric_data = pd.concat([numeric_data, test_data[['GENDER_1', 'ENROLLMENT STATUS_1']]], axis=1)
        numeric_data = numeric_data.drop(columns=['YEAR LEVEL', 'STUDENT NO'], errors='ignore')
        corr_matrix = numeric_data.corr()

        # Manually set display labels
        display_labels = corr_matrix.columns.tolist()
        # Replace specific labels for display purposes
        display_labels = [label.replace('GENDER_1', 'GENDER').replace('ENROLLMENT STATUS_1', 'ENROLLMENT STATUS') for label in display_labels]

        # Plot the heatmap
        plt.figure(figsize=(8, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", xticklabels=display_labels, yticklabels=display_labels)

        # Rotate x-axis labels for better visibility
        plt.xticks(rotation=45)

        # Save the plot
        img_corr_heatmap = io.BytesIO()
        plt.savefig(img_corr_heatmap, format='png', bbox_inches='tight')
        img_corr_heatmap.seek(0)
        correlation_heatmap_plot = base64.b64encode(img_corr_heatmap.getvalue()).decode('utf8')
        plt.close()

        # Retention by Course Plot
        if 'COURSE' in test_data.columns:
            courses = test_data['COURSE'].unique()
            course_labels = []
            retention_rates = []
            for course in courses:
                course_data = test_data[test_data['COURSE'] == course]
                if len(course_data) < 5:
                    continue
                retention_rate = course_data['RETAINED'].mean() * 100
                course_labels.append(course)
                retention_rates.append(retention_rate)

            plt.figure(figsize=(8, 6))
            bars = plt.bar(course_labels, retention_rates, color=[course_colors.get(course, '#333333') for course in course_labels])
            for bar, rate, label in zip(bars, retention_rates, course_labels):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f'{rate:.1f}%', ha='center', va='bottom', fontsize=14)
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, label, ha='center', va='center', rotation='vertical', fontsize=16, color='white')
           
            plt.xticks([])
            img_retention_rate = io.BytesIO()
            plt.savefig(img_retention_rate, format='png')
            img_retention_rate.seek(0)
            retention_by_course_plot = base64.b64encode(img_retention_rate.getvalue()).decode('utf8')
            plt.close()

        # Accuracy by Course Plot
        retained_accuracy = []
        course_labels_accuracy = []
        if 'COURSE' in test_data.columns:
            for course in courses:
                course_data = test_data[test_data['COURSE'] == course]
                if len(course_data) < 5:
                    continue
                X_course = imputer.transform(course_data[feature_names])
                y_course = course_data['RETAINED']
                y_pred_course = model.predict(X_course)
                accuracy_retained = accuracy_score(y_course, y_pred_course)
                retained_accuracy.append(accuracy_retained * 100)
                course_labels_accuracy.append(course)

            plt.figure(figsize=(8, 6))
            bars = plt.bar(course_labels_accuracy, retained_accuracy, color=[course_colors.get(course, '#333333') for course in course_labels_accuracy])
            for bar, accuracy, label in zip(bars, retained_accuracy, course_labels_accuracy):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f'{accuracy:.1f}%', ha='center', va='bottom', fontsize=14)
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, label, ha='center', va='center', rotation='vertical', fontsize=16, color='white')
            
            plt.xticks([])
            img_accuracy_rate = io.BytesIO()
            plt.savefig(img_accuracy_rate, format='png')
            img_accuracy_rate.seek(0)
            accuracy_by_course_plot = base64.b64encode(img_accuracy_rate.getvalue()).decode('utf8')
            plt.close()

        # Assuming y_test_pred is your predictions array, and 'RETAINED' is the actual values in the test dataset
        test_data['Predicted_Retention'] = y_test_pred
        
        # Calculate percentage retention for each student
        test_data['Predicted_Retention_Percentage'] = np.where(test_data['Predicted_Retention'] == 1, 100, 0)

        # If real retention data exists, calculate the actual retention percentage
        if 'RETAINED' in test_data.columns:
            test_data['Actual_Retention_Percentage'] = np.where(test_data['RETAINED'] == 1, 100, 0)
        else:
            test_data['Actual_Retention_Percentage'] = None  # In case the 'RETAINED' column doesn't exist

        # Add course-level summary percentages (e.g., retention rate per course)
        if 'COURSE' in test_data.columns:
            course_groups = test_data.groupby('COURSE')
            course_summary = course_groups['Predicted_Retention'].mean() * 100
            test_data = test_data.merge(course_summary.rename('Course_Retention_Rate'), on='COURSE', how='left')

        # Save the predictions and percentages to a CSV file
        result_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"predictions_{file.filename.split('.')[0]}.csv")
        test_data.to_csv(result_file_path, index=False)  # Save the prediction results as CSV

        if 'user_id' in session:
            result_entry = UserResults(user_id=session['user_id'], result_file=result_file_path, timestamp=datetime.utcnow())
            db.session.add(result_entry)
            db.session.commit()
        
        if 'user_id' in session:
            metrics_entry = UserMetrics(
                user_id=session['user_id'],
                feature_importance_plot=feature_importance_plot,
                correlation_heatmap_plot=correlation_heatmap_plot,
                retention_by_course_plot=retention_by_course_plot,
                accuracy_by_course_plot=accuracy_by_course_plot,
                timestamp=datetime.utcnow()
            )
            db.session.add(metrics_entry)
            db.session.commit()

        # Log the upload activity
        if 'user_id' in session:
            # Log to UploadHistory
            upload_entry = UploadHistory(user_id=session['user_id'], file_name=file.filename, timestamp=datetime.utcnow())
            db.session.add(upload_entry)
            db.session.commit()


        # Log the filename in session for later download
        session['result_file_path'] = result_file_path

        # Render the output template
        return render_template(
            'output.html',
            retention_by_course_summary=retention_by_course_summary,
            retention_prediction=retention_prediction,
            feature_importance=feature_importance_plot,
            correlation_heatmap=correlation_heatmap_plot,
            retention_by_course=retention_by_course_plot,
            accuracy_by_course=accuracy_by_course_plot,
            predicted_retention_rate=predicted_retention_rate,
            real_retention_rate=real_retention_rate,
            real_retained_students=real_retained_students,
            prediction_error=prediction_error,
            school_year=school_year,
            institute=institute,
            total_students=total_students,
            total_courses=total_courses,
            predicted_retained_students=predicted_retained_students,
            result_file_name=os.path.basename(result_file_path)
        )

    return render_template('index.html')

@app.route('/metricspage', methods=['GET'])
def metricspage():
    if 'user_id' not in session:
        flash("Please log in to view metrics.", "warning")
        return redirect(url_for('login'))
    
    user_id = session.get('user_id')
    
    user_id = session['user_id']
    user_metrics = UserMetrics.query.filter_by(user_id=user_id).order_by(UserMetrics.timestamp.desc()).first()

    # Debugging: Check if metrics were found
    if not user_metrics:
        print(f"No metrics found for user {user_id}.")
        flash("No metrics available for this user.", "warning")
        return redirect(url_for('homepage'))

    # If metrics are found, render the metricspage.html
    feature_importance_plot = user_metrics.feature_importance_plot
    correlation_heatmap_plot = user_metrics.correlation_heatmap_plot
    retention_by_course_plot = user_metrics.retention_by_course_plot
    accuracy_by_course_plot = user_metrics.accuracy_by_course_plot

    return render_template(
        'metricspage.html',
        feature_importance=feature_importance_plot,
        correlation_heatmap=correlation_heatmap_plot,
        retention_by_course=retention_by_course_plot,
        accuracy_by_course=accuracy_by_course_plot
    )

@app.route('/output', methods=['GET'])
def output():
    # Check if the user is logged in
    if 'user_id' not in session:
        flash("Please log in to view results.", "warning")
        return redirect(url_for('login'))

    # Get the logged-in user's results
    user_id = session['user_id']
    result_entry = UserResults.query.filter_by(user_id=user_id).order_by(UserResults.timestamp.desc()).first()

    if result_entry:
        return send_from_directory(app.config['UPLOAD_FOLDER'], os.path.basename(result_entry.result_file))
    else:
        flash("No saved results available for this user.", "warning")
        return redirect(url_for('homepage'))

@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        # Check if the user exists in the database
        user = User.query.filter_by(email=email).first()
        
        # If user exists, check password
        if user and check_password_hash(user.password, password):
            session['email'] = email
            session['user_id'] = user.id
            session['role'] = user.role  # Store the role in the session

            # Log the login activity
            activity = ActivityLog(user_id=user.id, activity="login", timestamp=datetime.utcnow())
            db.session.add(activity)
            db.session.commit()

            flash('Login successful!', 'success')

            # Redirect admin users to a special admin page
            if user.role == 'admin':
                return redirect(url_for('homepage'))
            else:
                return redirect(url_for('login'))
        else:
            flash('Invalid credentials, please try again.', 'danger')
            return redirect(url_for('login'))
    
    return render_template('login.html')


@app.route('/register', methods=['POST', 'GET'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        # Check if the email already exists
        if User.query.filter_by(email=email).first():
            flash('Email already exists!', 'danger')
            return redirect(url_for('register'))
        
        # Check if the email contains "@admin" to allow registration
        if "@admin.hau.edu.ph" in email:
            role = 'admin'
        else:
            flash('You are not allowed to register. Only admins are permitted.', 'danger')
            return redirect(url_for('login'))

        # Hash the password using pbkdf2:sha256
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(email=email, password=hashed_password, role=role)  # Set the role for the user
        
        db.session.add(new_user)
        db.session.commit()

        flash('Admin user registered successfully!', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/activity_logs')
def activity_logs():
    if 'role' in session and session['role'] == 'admin':
        # Query all activity logs
        logs = ActivityLog.query.order_by(ActivityLog.timestamp.desc()).all()
        return render_template('activity_logs.html', logs=logs)
    else:
        flash('Access denied. Admins only.', 'danger')
        return redirect(url_for('login'))

@app.route('/admin_dashboard')
def admin_dashboard():
    if 'email' not in session or session.get('role') != 'admin':
        flash('Access denied. Admins only.', 'danger')
        return redirect(url_for('login'))

    return render_template('homepage')  # Admin dashboard template

@app.route('/logout')
def logout():
    if 'user_id' in session:
        activity = ActivityLog(user_id=session['user_id'], activity="logout", timestamp=datetime.utcnow())
        db.session.add(activity)
        db.session.commit()
        
    session.pop('email', None)
    flash('You have been logged out!', 'info')
    return redirect(url_for('login'))

@app.route('/download/<filename>')
def download(filename):
    if 'email' not in session:
        flash('Please log in to access this page.', 'warning')
        return redirect(url_for('login'))

    # Log the download activity
    if 'user_id' in session:
        utc_timestamp = datetime.now(pytz.UTC)  # Generate timezone-aware UTC timestamp
        activity = DownloadHistory(user_id=session['user_id'], file_name=filename, timestamp=utc_timestamp)
        db.session.add(activity)
        db.session.commit()  # Make sure you commit the changes

    # Send the file for download
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/history')
def history():
    if 'email' not in session:
        flash('Please log in to access this page.', 'warning')
        return redirect(url_for('login'))

    # Get the current user's upload history from the database
    user_id = session['user_id']
    upload_history = UploadHistory.query.filter_by(user_id=user_id).order_by(UploadHistory.timestamp.desc()).all()

    return render_template('history.html', upload_history=upload_history)

@app.route('/save_progress', methods=['POST'])
def save_progress():
    # Get form data from request and store in session or database
    form_data = request.form.to_dict()
    session['progress'] = form_data  # Example: Storing in Flask session
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True)
