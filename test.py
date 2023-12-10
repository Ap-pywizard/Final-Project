#Attempt 1

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils import resample

# Step 1: Read in Data and Check Dimensions
file_path = '/Users/adampiro/Downloads/social_media_usage.csv'
s = pd.read_csv(file_path)
print(s.shape)

# Step 2: Define clean_sm Function and Test
def clean_sm(x):
    return np.where(x == 1, 1, 0)

# Testing the function
test_df = pd.DataFrame({'A': [1, 2, 1], 'B': [3, 1, 4]})
test_df = test_df.applymap(clean_sm)
print(test_df)

# Step 3: Create New DataFrame "ss"
ss = s[['web1h', 'income', 'educ2', 'par', 'marital', 'gender', 'age']].copy()
ss['sm_li'] = ss['web1h'].apply(clean_sm)
ss['income'] = np.where(ss['income'] > 9, np.nan, ss['income'])
ss['education'] = np.where(ss['educ2'] > 8, np.nan, ss['educ2'])
ss['parent'] = np.where(ss['par'] == 1, 1, 0)
ss['married'] = np.where(ss['marital'] == 1, 1, 0)
ss['female'] = np.where(ss['gender'] == 2, 1, 0)
ss['male'] = np.where(ss['gender'] == 1, 1, 0)  # Assuming '1' represents male in the 'gender' column
# ... (rest of your existing code) ...
ss['age'] = np.where(ss['age'] > 98, np.nan, ss['age'])
ss.dropna(inplace=True)
print(ss.head())

# Step 4: Drop Any Missing Values and Perform Exploratory Analysis
# Exploratory analysis code goes here (not provided in this snippet)

# Step 5: Create Target Vector and Feature Set
y = ss['sm_li']
X = ss.drop(['sm_li', 'web1h', 'gender', 'educ2', 'marital', 'par'], axis=1)

# Step 6: Split Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Step 7: Instantiate Logistic Regression Model and Fit with Training Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
logreg = LogisticRegression(class_weight='balanced', max_iter=5000)
logreg.fit(X_train_scaled, y_train)

# Step 8: Evaluate the Model Using the Testing Data
y_pred = logreg.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Step 9: Generate a Confusion Matrix and Interpret
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Step 10: Create Confusion Matrix DataFrame
conf_matrix_df = pd.DataFrame(conf_matrix, 
                              index=['Actual Non-User', 'Actual User'], 
                              columns=['Predicted Non-User', 'Predicted User'])
print(conf_matrix_df)

# Step 11: Calculate Precision, Recall, and F1 Score by Hand
TP = conf_matrix[1, 1]  # True Positives
TN = conf_matrix[0, 0]  # True Negatives
FP = conf_matrix[0, 1]  # False Positives
FN = conf_matrix[1, 0]  # False Negatives

precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * (precision * recall) / (precision + recall)
print("Precision:", precision, "Recall:", recall, "F1 Score:", f1_score)

# Step 12: Classification Report
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)

# Scenario 1: High-income (8), highly educated (7), non-parent, married female (1) who is 42 years old
scenario_1_original = pd.DataFrame({
    'income': [10],
    'age': [35], 
    'education': [7],
    'parent': [0],
    'married': [1],
    'female': [1],
    'male': [1],
})

# Generate polynomial features for Scenario 1
poly = PolynomialFeatures(degree=1, include_bias=False)
poly.fit(X_train[["income","age", "education" ]])
#scenario_1_poly = poly.transform(scenario_1_original[['income', 'education', 'age']])
#scenario_1_poly_df = pd.DataFrame(scenario_1_poly, columns=poly.get_feature_names_out(['income', 'education', 'age']))

# Combine original features with polynomial features
#scenario_1_combined = pd.concat([scenario_1_original.drop(['income', 'education', 'age'], axis=1), scenario_1_poly_df], axis=1)

# Make sure to include all features in the correct order as used in the model training
# Add any missing columns with default values
#for col in X_train.columns:
 #   if col not in scenario_1_combined:
  #      scenario_1_combined[col] = 0

# Reorder columns to match training data
#scenario_1_combined = scenario_1_combined[X_train.columns]

# Scale the features
#scenario_1_scaled = scaler.transform(scenario_1_combined)

# Make prediction
#pred_scenario_1 = logreg.predict_proba(scenario_1_scaled)[0][1]
#print("Probability for Scenario 1:", pred_scenario_1)


#Streamlit application deployment
def make_prediction(input_data):
    
    poly_features = poly.transform(input_data[["income","age", "education" ]])
    poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(["income","age", "education"]))
    combined = pd.concat([input_data.drop(["income","age", "education" ], axis=1), poly_df], axis=1)

    for col in X_train.columns:
        if col not in combined.columns:
            combined[col] = 0

    combined = combined[X_train.columns]

    scaled_features = scaler.transform(combined)

    probability = logreg.predict_proba(scaled_features)[0][1]

    return probability






import streamlit as st


# Streamlit interface
st.title("LinkedIn User Prediction")

with st.form("user_input_form"):
    income = st.number_input("Income Level (1-9)", min_value=1, max_value=9, value=1)
    age = st.number_input("Age", min_value=1, max_value=100, value=25)
    education = st.number_input("Education Level (1-8)", min_value=1, max_value=8, value=1)
    parent = st.selectbox("Are you a parent? (1 for Yes, 0 for No)", options=[1, 0])
    married = st.selectbox("Are you married? (1 for Yes, 0 for No)", options=[1, 0])
    gender = st.selectbox("Gender (1 for Male, 2 for Female)", options=[1, 2])
   

    submit_button = st.form_submit_button("Predict LinkedIn Usage")

if submit_button:
    
    # Convert gender to 'female' and 'male' binary columns
    female = 1 if gender == 2 else 0
    male = 1 if gender == 1 else 0

    # Prepare the input data and make prediction
    input_data = pd.DataFrame(
        [[income, age, education, parent, married, female, male]],
        columns=['income','age', 'education','parent','married','female','male']
    )


    prediction = make_prediction(input_data)

    # Display the result
    st.write(f"Probability of being a LinkedIn user: {prediction:.2f}")
    if prediction >= 0.5 :
        st.write(f"Yes, this is a Linked In user!")
    else :
        st.write(f"No, this is not a Linked In user!")
#run application








































































