#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib  # For saving and loading models

# Function to preprocess data
def preprocess_data(data):
    categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    label_encoders = {col: LabelEncoder() for col in categorical_cols}
    
    # Encode categorical variables
    for col in categorical_cols:
        data[col] = label_encoders[col].fit_transform(data[col].str.strip().str.capitalize())
    
    return data, label_encoders

# Function to train and save the model
def train_and_save_model(data, model_path, encoders_path):
    # Preprocess data
    data, label_encoders = preprocess_data(data)
    
    # Separate features and target
    X = data.drop(columns=['stroke'])
    y = data['stroke']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Train the model
    rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
    rf_model.fit(X_train_resampled, y_train_resampled)
    
    # Save the model and encoders
    joblib.dump(rf_model, model_path)
    joblib.dump(label_encoders, encoders_path)
    
    # Evaluate the model
    y_pred = rf_model.predict(X_test)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nAccuracy:", accuracy_score(y_test, y_pred))

# Function to load the saved model and encoders
def load_model_and_encoders(model_path, encoders_path):
    rf_model = joblib.load(model_path)
    label_encoders = joblib.load(encoders_path)
    return rf_model, label_encoders

# Function to predict stroke risk based on user input
def predict_stroke(user_data, rf_model, label_encoders):
    # Encode user input
    categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    try:
        for col in categorical_cols:
            user_data[col] = label_encoders[col].transform([user_data[col]])[0]
        
        # Convert user data to DataFrame
        user_df = pd.DataFrame([user_data])
        
        # Make prediction
        user_prediction = rf_model.predict(user_df)[0]
        user_prediction_prob = rf_model.predict_proba(user_df)[0][1]
        
        # Display the result
        if user_prediction == 1:
            print(f"\nThe patient is predicted to have a stroke risk with a probability of {user_prediction_prob:.2f}.")
        else:
            print(f"\nThe patient is predicted to not have a stroke risk with a probability of {user_prediction_prob:.2f}.")
    except ValueError as e:
        print("\nError: Invalid input. Please ensure all inputs are correct.")

# Main program flow
if __name__ == "__main__":
    model_path = "rf_model.pkl"
    encoders_path = "label_encoders.pkl"
    
    # Load dataset
    data = pd.read_csv('full_data.csv')
    
    # Check if model and encoders exist, otherwise train and save them
    try:
        rf_model, label_encoders = load_model_and_encoders(model_path, encoders_path)
        print("Model and LabelEncoders loaded successfully!")
    except:
        print("Training the model...")
        train_and_save_model(data, model_path, encoders_path)
        rf_model, label_encoders = load_model_and_encoders(model_path, encoders_path)
    
    # Take user input
    print("\nEnter patient details to predict stroke:")
    user_data = {
        'gender': input("Gender (Male/Female): ").strip().capitalize(),
        'age': float(input("Age: ").strip()),
        'hypertension': int(input("Hypertension (0/1): ").strip()),
        'heart_disease': int(input("Heart Disease (0/1): ").strip()),
        'ever_married': input("Ever Married (Yes/No): ").strip().capitalize(),
        'work_type': input("Work Type (Private/Self-employed/Govt_job/Children/Never_worked): ").strip().capitalize(),
        'Residence_type': input("Residence Type (Urban/Rural): ").strip().capitalize(),
        'avg_glucose_level': float(input("Average Glucose Level: ").strip()),
        'bmi': float(input("BMI: ").strip()),
        'smoking_status': input("Smoking Status (Never smoked/Formerly smoked/Smokes/Unknown): ").strip().capitalize()
    }
    
    # Predict stroke risk
    predict_stroke(user_data, rf_model, label_encoders)


# In[ ]:




