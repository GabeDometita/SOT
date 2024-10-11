import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import hashlib
import os
import optuna
from imblearn.over_sampling import ADASYN
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# Function encapsulating the entire process
def train_model(filepath):
    # Create necessary directories
    os.makedirs('models', exist_ok=True)

    def load_data(filepath):
        excel_file = pd.ExcelFile(filepath)
        sheet_names = excel_file.sheet_names  # Dynamically get all sheet names

        all_data = pd.concat(
            [pd.read_excel(filepath, sheet_name=sheet).assign(SheetName=sheet) for sheet in sheet_names]
        )
        all_data = all_data.reset_index(drop=True)

        return all_data

    # Load the data
    all_data = load_data(filepath)

    # Drop irrelevant columns
    all_data.drop(columns=['STUDENT NO', 'STUDENT NAME', 'SheetName'], inplace=True)

    # Create and save dataset metadata
    dataset_metadata = {
        'num_rows': len(all_data),
        'num_columns': len(all_data.columns),
        'column_names': list(all_data.columns),
        'data_hash': hashlib.md5(pd.util.hash_pandas_object(all_data, index=False).values).hexdigest()
    }
    for col in all_data.columns:
        dataset_metadata[f'{col}_hash'] = hashlib.md5(pd.util.hash_pandas_object(all_data[col], index=False).values).hexdigest()

    # Save metadata
    joblib.dump(dataset_metadata, 'models/dataset_metadata.pkl')

    # Handle missing values
    all_data['GPA'] = pd.to_numeric(all_data['GPA'], errors='coerce')
    imputer = SimpleImputer(strategy='mean')
    all_data[['GPA', 'PASSED', 'FAILED']] = imputer.fit_transform(all_data[['GPA', 'PASSED', 'FAILED']])

    # Convert GENDER and ENROLLMENT STATUS to categorical and one-hot encode them
    all_data['GENDER'] = all_data['GENDER'].astype('category')
    all_data['ENROLLMENT STATUS'] = all_data['ENROLLMENT STATUS'].astype('category')
    all_data = pd.get_dummies(all_data, columns=['GENDER', 'ENROLLMENT STATUS'], drop_first=True)

    # Standardize date format
    all_data['BIRTH DATE'] = pd.to_datetime(all_data['BIRTH DATE']).dt.strftime('%Y-%m-%d')

    # Round float columns
    float_columns = all_data.select_dtypes(include=['float64']).columns
    all_data[float_columns] = all_data[float_columns].round(6)

    # Select predictors
    predictors = ['PASSED', 'FAILED', 'GPA', 'AGE'] + list(all_data.columns[all_data.columns.str.startswith('GENDER')]) + list(all_data.columns[all_data.columns.str.startswith('ENROLLMENT STATUS')])

    X = all_data[predictors]
    y = all_data['RETAINED']

    # Handle missing values for predictors
    X = imputer.fit_transform(X)

    # Apply ADASYN to balance the classes
    adasyn = ADASYN(random_state=42)
    X_resampled, y_resampled = adasyn.fit_resample(X, y)

    # Save the resampled data to metadata
    dataset_metadata['num_resampled'] = len(X_resampled)
    joblib.dump(dataset_metadata, 'models/dataset_metadata.pkl')

    # Bayesian Optimization with Optuna
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'lambda': trial.suggest_float('lambda', 0.01, 50.0),
            'alpha': trial.suggest_float('alpha', 0.01, 50.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
        }

        model = XGBClassifier(**params)

        # Use 5-fold cross-validation
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []

        # Perform cross-validation manually
        for fold, (train_idx, test_idx) in enumerate(skf.split(X_resampled, y_resampled), 1):
            X_train, X_test = X_resampled[train_idx], X_resampled[test_idx]
            y_train, y_test = y_resampled[train_idx], y_resampled[test_idx]

            eval_set = [(X_test, y_test)]
            model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

            y_pred = model.predict(X_test)

            # Calculate metrics for each fold
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            # Append to corresponding lists
            accuracy_scores.append(accuracy)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            
        # After cross-validation, print the mean of the metrics
        print(f"Mean Accuracy: {np.mean(accuracy_scores):.4f}")
        print(f"Mean Precision: {np.mean(precision_scores):.4f}")
        print(f"Mean Recall: {np.mean(recall_scores):.4f}")
        print(f"Mean F1 Score: {np.mean(f1_scores):.4f}")

        return np.mean(accuracy_scores)

    # Create Optuna study and optimize
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    # Get the best hyperparameters
    best_params = study.best_params
    print(f"Best hyperparameters: {best_params}")

    # Train the final model with the best parameters
    model = XGBClassifier(**best_params)
    model.fit(X_resampled, y_resampled)

    # Save the best parameters and model
    dataset_metadata['best_params'] = best_params
    joblib.dump(model, 'models/pretrained_model_xgboost.pkl')
    joblib.dump(list(predictors), 'models/feature_names.pkl')
    joblib.dump(imputer, 'models/imputer.pkl')

    # Feature Importance
    feature_importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': predictors,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False)

    # Save the feature importance to metadata
    dataset_metadata['feature_importance'] = feature_importance_df.to_dict()

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title('Feature Importance')
    # plt.show()

    # Save the feature importance plot
    plt.savefig('models/feature_importance.png')

    # Ensure only numeric columns are used for correlation
    numeric_data = all_data.select_dtypes(include=['float64', 'int64'])

    # Compute the correlation matrix on numeric data only
    corr_matrix = numeric_data.corr()

    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    # plt.show()

    # Save the heatmap to a file
    plt.savefig('models/correlation_heatmap.png')

    ### RETENTION RATE BY COURSE PLOT ###
    # Calculate retention rates for each course
    courses = all_data['COURSE'].unique()
    course_labels = []
    retention_rates = []

    for course in courses:
        course_data = all_data[all_data['COURSE'] == course]
        if len(course_data) < 5:
            continue
        retention_rate = course_data['RETAINED'].mean() * 100
        course_labels.append(course)
        retention_rates.append(retention_rate)

    # Plot retention rates by course
    plt.figure(figsize=(12, 6))
    bars = plt.bar(course_labels, retention_rates, color='skyblue')
    plt.xlabel('Course')
    plt.ylabel('Retention Rate (%)')
    plt.title('Retention Rate by Course')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 100)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.1f}%', ha='center', va='bottom')

    # Save retention plot to a PNG image and encode to base64
    plt.savefig('models/retention_by_course.png')

    ### ACCURACY RATE RETENTION PREDICTION BY COURSE ###
    # Predict retention per course and compute accuracy for each
    retained_accuracy = []
    course_labels = []

    for course in courses:
        course_data = all_data[all_data['COURSE'] == course]
        if len(course_data) < 5:
            continue
        X_course = imputer.transform(course_data[predictors])
        y_course = course_data['RETAINED']
        y_pred = model.predict(X_course)

        accuracy_retained = accuracy_score(y_course, y_pred)
        retained = (y_course == 1).sum()

        if retained > 0:
            retained_accuracy.append(accuracy_retained)
            course_labels.append(course)

    # Plot accuracy rates for retained students
    x = np.arange(len(course_labels))  # Label locations
    width = 0.35  # Width of bars

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x, retained_accuracy, width, label='Retained', color='blue')

    # Add labels, title, and format
    ax.set_xlabel('Course')
    ax.set_ylabel('Accuracy Rate')
    ax.set_title('Accuracy Rate for Student Retention by Course')
    ax.set_xticks(x)
    ax.set_xticklabels(course_labels, rotation=45, ha='right')

    # Add accuracy rate values on top of bars
    for bar in bars1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}', ha='center', va='bottom', color='black')

    # Adjust layout
    plt.tight_layout()

    # Save accuracy rate plot
    plt.savefig('models/accuracy_rate_by_course.png')


    print("Model, plots, and metadata saved successfully.")

# Call the train_model function
train_model('uploads/FINAL-TRAINING-SEALIST.xlsx')
