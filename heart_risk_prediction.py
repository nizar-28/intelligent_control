import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tkinter import filedialog, messagebox
import tkinter as tk
import os
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
import xgboost as xgb

data = None
model = None
X_test_scaled = None
scaler = StandardScaler()
X_train = None  # Initialize X_train as a global variable

# Load the dataset
def load_dataset():
    global data
    file_path = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select file",
                                           filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
    if file_path:
        data = pd.read_csv(file_path)
        messagebox.showinfo("Success", "Dataset loaded successfully!")

# Train the model
def train_model():
    global model, X_train_scaled, y_train, grid_search, X_train  # Assign X_train to the global variable
    if data is None:
        messagebox.showerror("Error", "Please load the dataset first!")
        return
    
    # Extract numerical features
    blood_pressure = data['Blood Pressure'].str.split('/', expand=True)
    X = data.drop(columns=['Patient ID'])
    X['Systolic_BP'] = blood_pressure[0].astype(float)
    X['Diastolic_BP'] = blood_pressure[1].astype(float)
    
    # Convert categorical variables into numerical format
    label_encoder = LabelEncoder()
    X['Sex'] = label_encoder.fit_transform(X['Sex'])
    X['Diet'] = label_encoder.fit_transform(X['Diet'])
    X['Country'] = label_encoder.fit_transform(X['Country'])
    X['Continent'] = label_encoder.fit_transform(X['Continent'])
    X['Hemisphere'] = label_encoder.fit_transform(X['Hemisphere'])
    
    y = data['Heart Attack Risk']
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize the data
    X_train_scaled = scaler.fit_transform(X_train.drop(columns=['Heart Attack Risk', 'Blood Pressure']))  # Exclude target variable and 'Blood Pressure' from scaling
    
    # Oversample minority class using SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    # Define parameter grid for RandomizedSearchCV
    param_grid_xgb = {
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.5, 0.7, 0.9],
        'colsample_bytree': [0.5, 0.7, 0.9],
        'n_estimators': [100, 200, 300]
    }
    
    # Perform RandomizedSearchCV with XGBoost
    xgb_clf = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
    random_search_xgb = RandomizedSearchCV(estimator=xgb_clf, param_distributions=param_grid_xgb, n_iter=100,
                                           scoring='accuracy', n_jobs=-1, cv=3, random_state=42, verbose=2)
    random_search_xgb.fit(X_train_resampled, y_train_resampled)
    
    # Get the best parameters
    best_params_xgb = random_search_xgb.best_params_
    
    # Train the XGBoostClassifier with the best parameters
    xgb_model = xgb.XGBClassifier(objective='binary:logistic', random_state=42, **best_params_xgb)
    xgb_model.fit(X_train_resampled, y_train_resampled)
    
    # Set the trained model as the final model
    model = xgb_model
    
    # Evaluate the XGBoostClassifier on the training data
    print("Model Performance on Training Data (XGBoost):")
    y_train_pred_xgb = xgb_model.predict(X_train_resampled)
    print(classification_report(y_train_resampled, y_train_pred_xgb))
    
    messagebox.showinfo("Success", "Model trained successfully!")

# Evaluate the model
def evaluate_model():
    global model, y_test, X_test, X_train
    if model is None:
        messagebox.showerror("Error", "Please train the model first!")
        return
    file_path = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select file",
                                           filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
    if file_path:
        test_data = pd.read_csv(file_path)
        # Check if 'Heart Attack Risk' column exists in the test data
        if 'Heart Attack Risk' not in test_data.columns:
            messagebox.showerror("Error", "Test data must include 'Heart Attack Risk' column!")
            return
        # Preprocess 'Blood Pressure' column in the test data
        test_data['Systolic_BP'] = test_data['Blood Pressure'].apply(lambda x: float(x.split('/')[0]))
        test_data['Diastolic_BP'] = test_data['Blood Pressure'].apply(lambda x: float(x.split('/')[1]))
        # Drop unnecessary columns
        test_data = test_data.drop(columns=['Patient ID', 'Blood Pressure'])
        # Convert categorical variables into numerical format
        label_encoder = LabelEncoder()
        for column in ['Sex', 'Diet', 'Country', 'Continent', 'Hemisphere']:
            test_data[column] = label_encoder.fit_transform(test_data[column])
        # Extract y_test
        y_test = test_data['Heart Attack Risk']
        # Drop the target variable for imputation
        X_test_imputed = test_data.drop(columns=['Heart Attack Risk'])
        # Normalize the test data using the mean and standard deviation from the training data
        X_test_scaled = scaler.transform(X_test_imputed)
        # Predict using the best model
        y_pred_best = model.predict(X_test_scaled)
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred_best)
        
        # Calculate precision, recall, and F1-score for each class
        class_names = np.unique(y_test)
        precision = precision_score(y_test, y_pred_best, average=None, labels=class_names)
        recall = recall_score(y_test, y_pred_best, average=None, labels=class_names)
        f1 = f1_score(y_test, y_pred_best, average=None, labels=class_names)
        
        # Calculate overall average precision, recall, and F1-score
        avg_precision = precision_score(y_test, y_pred_best, average='weighted')
        avg_recall = recall_score(y_test, y_pred_best, average='weighted')
        avg_f1 = f1_score(y_test, y_pred_best, average='weighted')
        
        # Construct a classification report
        report = classification_report(y_test, y_pred_best)
        
        # Display results
        result_message = f"Accuracy: {accuracy}\n\n"
        result_message += "Precision, Recall, and F1-score for each class:\n\n"
        for i, class_name in enumerate(class_names):
            result_message += f"Class {class_name}:\n"
            result_message += f"Precision: {precision[i]}\n"
            result_message += f"Recall: {recall[i]}\n"
            result_message += f"F1-score: {f1[i]}\n\n"
        result_message += f"Average Precision: {avg_precision}\n"
        result_message += f"Average Recall: {avg_recall}\n"
        result_message += f"Average F1-score: {avg_f1}\n\n"
        result_message += "Classification Report:\n\n"
        result_message += report
        
        messagebox.showinfo("Model Evaluation", result_message)

# GUI setup
root = tk.Tk()
root.title("Heart Attack Risk Prediction")

# Buttons
load_button = tk.Button(root, text="Load Dataset", command=load_dataset)
load_button.pack(pady=10)

train_button = tk.Button(root, text="Train Model", command=train_model)
train_button.pack(pady=5)

evaluate_button = tk.Button(root, text="Evaluate Model", command=evaluate_model)
evaluate_button.pack(pady=5)

root.mainloop()
