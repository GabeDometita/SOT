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
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from datetime import datetime
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

# Define the User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

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

def generate_heatmap_interpretation(corr_matrix):
    """
    Categorizes and lists correlations into Positive, Neutral, and Negative based on the thresholds.
    Positive: Correlations ≥ 0.10
    Negative: Correlations ≤ -0.10
    Neutral: Correlations between -0.10 and 0.10 (exclusive).
    Returns the interpretation in descending order of correlation strength.
    """
    positive_correlations = []
    neutral_correlations = []
    negative_correlations = []

    # Iterate over the correlation matrix
    for row in corr_matrix.index:
        for col in corr_matrix.columns:
            if row != col:  # Skip diagonal (self-correlation)
                corr_value = corr_matrix.at[row, col]

                # Categorize correlations based on thresholds
                if corr_value >= 0.10:  # Positive correlation
                    positive_correlations.append((row, col, corr_value))
                elif corr_value <= -0.10:  # Negative correlation
                    negative_correlations.append((row, col, corr_value))
                else:  # Neutral correlation
                    neutral_correlations.append((row, col, corr_value))
    
    # Sort correlations by absolute value, in descending order
    positive_correlations.sort(key=lambda x: abs(x[2]), reverse=True)
    neutral_correlations.sort(key=lambda x: abs(x[2]), reverse=True)
    negative_correlations.sort(key=lambda x: abs(x[2]), reverse=True)

    return positive_correlations, neutral_correlations, negative_correlations

def interpret_correlation(feature1, feature2, correlation_value):
    """
    Generates a verbal interpretation of the correlation between two features.
    
    Parameters:
    - feature1: The name of the first feature (e.g., "RETAINED").
    - feature2: The name of the second feature (e.g., "PASSED").
    - correlation_value: The correlation value between the two features (e.g., 0.87).
    
    Returns:
    - A string that explains the meaning of the correlation.
    """
    abs_value = abs(correlation_value)
    
    # Determine the strength of the correlation
    if abs_value > 0.7:
        strength = "strong"
    elif abs_value > 0.3:
        strength = "moderate"
    elif abs_value >= 0.1:
        strength = "weak"
    else:
        strength = "no significant"

    # Determine the direction of the correlation
    if correlation_value > 0:
        direction = "positive"
        interpretation = f"There is {strength} {direction} correlation between {feature1} and {feature2}. As {feature1} increases, {feature2} also tends to increase."
    elif correlation_value < 0:
        direction = "negative"
        interpretation = f"There is {strength} {direction} correlation between {feature1} and {feature2}. As {feature1} increases, {feature2} tends to decrease."
    else:
        interpretation = f"There is no significant correlation between {feature1} and {feature2}."

    return interpretation


@app.route('/', methods=['GET', 'POST'])
def homepage():

    if 'username' not in session:
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

        institute = "SEA"

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
        heatmap_interpretation = generate_heatmap_interpretation(corr_matrix)

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

        # Store plots in session (server-side)
        session['feature_importance_plot'] = feature_importance_plot
        session['correlation_heatmap_plot'] = correlation_heatmap_plot
        session['retention_by_course_plot'] = retention_by_course_plot
        session['accuracy_by_course_plot'] = accuracy_by_course_plot
        session['heatmap_interpretation'] = heatmap_interpretation
        session['test_data'] = test_data.to_dict()  # Store test data as dictionary in the session



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
            heatmap_interpretation=heatmap_interpretation
        )

    return render_template('index.html')

@app.route('/metricspage', methods=['GET'])
def metricspage():
    # Retrieve plots from the session (server-side)
    feature_importance_plot = session.get('feature_importance_plot')
    correlation_heatmap_plot = session.get('correlation_heatmap_plot')
    retention_by_course_plot = session.get('retention_by_course_plot')
    accuracy_by_course_plot = session.get('accuracy_by_course_plot')

    # Assuming test_data is already stored in session from the homepage route
    test_data = session.get('test_data')

    if test_data is None:
        return "Test data is not available. Please upload a file first."

    # Convert test data back into a DataFrame
    test_data = pd.DataFrame(test_data)

    # Preprocess test data for correlation calculation
    numeric_data = test_data.select_dtypes(include=[np.float64, np.int64, np.uint8])
    if 'GENDER_1' in test_data.columns and 'ENROLLMENT STATUS_1' in test_data.columns:
        numeric_data = pd.concat([numeric_data, test_data[['GENDER_1', 'ENROLLMENT STATUS_1']]], axis=1)
    numeric_data = numeric_data.drop(columns=['YEAR LEVEL', 'STUDENT NO'], errors='ignore')
    
    # Calculate correlation matrix
    corr_matrix = numeric_data.corr()

    # Generate categorized interpretations (positive, neutral, negative)
    positive_correlations, neutral_correlations, negative_correlations = generate_heatmap_interpretation(corr_matrix)

    # Render the metricspage.html with the available plots and categorized correlations
    return render_template(
        'metricspage.html',
        feature_importance=feature_importance_plot,
        correlation_heatmap=correlation_heatmap_plot,
        retention_by_course=retention_by_course_plot,
        accuracy_by_course=accuracy_by_course_plot,
        positive_correlations=positive_correlations,
        neutral_correlations=neutral_correlations,
        negative_correlations=negative_correlations
    )

@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Check if the user exists in the database
        user = User.query.filter_by(username=username).first()
        
        # If user exists, check password
        if user and check_password_hash(user.password, password):
            session['username'] = username
            session['user_id'] = user.id

            # Log the login activity
            activity = ActivityLog(user_id=user.id, activity="login", timestamp=datetime.utcnow())
            db.session.add(activity)
            db.session.commit()

            flash('Login successful!', 'success')
            return redirect(url_for('homepage'))
        else:
            flash('Invalid credentials, please try again.', 'danger')
            return redirect(url_for('login'))
    
    return render_template('login.html')

@app.route('/register', methods=['POST', 'GET'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Check if the username already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists!', 'danger')
            return redirect(url_for('register'))
        
        # Hash the password using pbkdf2:sha256
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, password=hashed_password)
        
        db.session.add(new_user)
        db.session.commit()

        print(f"New user registered: {username}")  # Debugging line
        
        flash('User registered successfully!', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    if 'user_id' in session:
        activity = ActivityLog(user_id=session['user_id'], activity="logout", timestamp=datetime.utcnow())
        db.session.add(activity)
        db.session.commit()
        
    session.pop('username', None)
    flash('You have been logged out!', 'info')
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
