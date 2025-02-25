

import pandas as pd

data = pd.read_csv(r"C:\\Users\\Vanshika Rana\\Downloads\\Mat Health\\Maternal Health Risk Data Set.csv")


# In[68]:


data.head(10)


# In[69]:


import matplotlib.pyplot as plt
import seaborn as sns

# Bar Plot for Risk Level Distribution

plt.figure(figsize=(8, 6))
sns.countplot(x='RiskLevel', data=data, palette="viridis")
plt.title('Risk Levels')
plt.xlabel('Risk Level')
plt.ylabel('Count')
plt.show()


# In[55]:


#Correlation Heatmap

import numpy as np

plt.figure(figsize=(8,6))
sns.heatmap(data.corr(), annot=True, cmap='viridis', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()


# In[56]:


#  Box Plot to Compare Age vs Risk Level

plt.figure(figsize=(8, 6))
sns.boxplot(x='RiskLevel', y='Age', data=data)
plt.title('Age Distribution across Risk Levels')
plt.xlabel('Risk Level')
plt.ylabel('Age')
plt.show()


# In[57]:


# Scatter Plot for HeartRate vs Age

plt.figure(figsize=(8, 6))
sns.scatterplot(x='Age', y='HeartRate', data=data, hue='RiskLevel', palette='Set1')
plt.title('Heart Rate vs Age with Risk Levels')
plt.xlabel('Age')
plt.ylabel('Heart Rate')
plt.show()


# In[58]:


# Check count of null values
null_values = data.isnull().sum()

print(null_values)


# In[70]:


from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import optuna
from xgboost import XGBClassifier

# Adding meaningful features
data['PulsePressure'] = data['SystolicBP'] - data['DiastolicBP']  # The difference between systolic and diastolic BP
data['MeanArterialPressure'] = data['DiastolicBP'] + (1/3) * (data['SystolicBP'] - data['DiastolicBP'])  # Mean Arterial Pressure
data['BloodPressureRatio'] = data['SystolicBP'] / data['DiastolicBP']  # Ratio of systolic to diastolic BP
data['HeartRateToAgeRatio'] = data['HeartRate'] / data['Age']  # Ratio of heart rate to age

# Encoding
encoder = LabelEncoder()
data['RiskLevelEncoded'] = encoder.fit_transform(data['RiskLevel'])

features = data.drop(columns=['RiskLevel', 'RiskLevelEncoded'])
target = data['RiskLevelEncoded']

# Convert all columns to numeric 
features = features.apply(pd.to_numeric, errors='coerce')

# Standardizing the features 
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Handling class imbalance
smote = SMOTE(random_state=42)
resampled_features, resampled_target = smote.fit_resample(scaled_features, target)

# Splitting the dataset into training and testing sets
train_features, test_features, train_target, test_target = train_test_split(
    resampled_features, resampled_target, test_size=0.2, random_state=42, stratify=resampled_target
)

def objective(trial):
    # hyperparameters for the XGBoost classifier
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 30, 500),  # Number of trees in the forest
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),  # Learning rate for the model
        'max_depth': trial.suggest_int('max_depth', 3, 10),  # Maximum depth of the trees
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),  # Fraction of samples used for training each tree
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),  # Fraction of features used for each tree
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 0.01, 10),  # L2 regularization term
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 0.01, 10),  # L1 regularization term
        'objective': 'multi:softmax',  # Multi-class classification objective
        'num_class': 3,  # Number of classes in the target
        'random_state': 25  # Ensuring reproducibility
    }

    # XGBoost classifier 
    model = XGBClassifier(**params)

    # Use cross-validation to evaluate the model performance
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(model, train_features, train_target, cv=cv, scoring='accuracy').mean()
    return score

# Optimize hyperparameters using Optuna
study = optuna.create_study(direction='maximize')  # We want to maximize accuracy
study.optimize(objective, n_trials=20)  # Run 20 trials for optimization

best_params = study.best_params
best_params['objective'] = 'multi:softmax'  # Multi-class classification
best_params['num_class'] = 3  # Three classes in the target variable
best_params['random_state'] = 42  # Ensuring reproducibility

# Train the XGBoost model 
final_model = XGBClassifier(**best_params)
final_model.fit(train_features, train_target)

# Evaluate the model on the test set
test_predictions = final_model.predict(test_features)
test_accuracy = accuracy_score(test_target, test_predictions)

# Displaying the results
print(f"Test Accuracy of the XGBoost Model: {test_accuracy:.4f}")
print("Classification Report:")
print(classification_report(test_target, test_predictions, target_names=encoder.classes_))


# In[25]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# Splitting the dataset into training and testing sets
train_features, test_features, train_target, test_target = train_test_split(
    resampled_features, resampled_target, test_size=0.2, random_state=42, stratify=resampled_target
)

# Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(train_features, train_target)

# Evaluating the model 
test_predictions = model.predict(test_features)
test_accuracy = accuracy_score(test_target, test_predictions)

# Display the results
print(f"Test Accuracy of the Logistic Regression Model: {test_accuracy:.4f}")
print("Classification Report:")
print(classification_report(test_target, test_predictions, target_names=encoder.classes_))


# In[47]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report

data['PulsePressure'] = data['SystolicBP'] - data['DiastolicBP']
data['MAP'] = data['DiastolicBP'] + (1/3) * (data['SystolicBP'] - data['DiastolicBP'])
data['BP_Ratio'] = data['SystolicBP'] / data['DiastolicBP']
data['HR_Age_Ratio'] = data['HeartRate'] / data['Age']

X = data.drop(columns=['RiskLevel', 'RiskLevelEncoded'])
y = data['RiskLevelEncoded']

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# DNN model
def create_dnn_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(3, activation='softmax')  # 3 classes: low, mid, high risk
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = create_dnn_model(X_train.shape[1])

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test),
                    callbacks=[early_stop, lr_scheduler], verbose=1)

# Evaluation
y_pred = np.argmax(model.predict(X_test), axis=1)
accuracy = accuracy_score(y_test, y_pred)

print(f"DNN Test Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))


# In[71]:


from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import optuna
from xgboost import XGBClassifier
from imblearn.under_sampling import TomekLinks
import pandas as pd

# Adding more meaningful features
data['PulsePressure'] = data['SystolicBP'] - data['DiastolicBP']
data['MeanArterialPressure'] = data['DiastolicBP'] + (1/3) * (data['SystolicBP'] - data['DiastolicBP'])
data['BloodPressureRatio'] = data['SystolicBP'] / data['DiastolicBP']
data['HeartRateToAgeRatio'] = data['HeartRate'] / data['Age']

# Encoding
encoder = LabelEncoder()
data['RiskLevelEncoded'] = encoder.fit_transform(data['RiskLevel'])

features = data.drop(columns=['RiskLevel', 'RiskLevelEncoded'])
target = data['RiskLevelEncoded']

# Convert all columns to numeric 
features = features.apply(pd.to_numeric, errors='coerce')

# Standardizing the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Handling class imbalance using SMOTE and TomekLinks for cleaner resampling
smote = SMOTE(random_state=42, sampling_strategy='auto')
resampled_features, resampled_target = smote.fit_resample(scaled_features, target)

# Apply Tomek Links after SMOTE to clean up the boundary between classes
tl = TomekLinks()
resampled_features, resampled_target = tl.fit_resample(resampled_features, resampled_target)

# Splitting the dataset into training and testing sets
train_features, test_features, train_target, test_target = train_test_split(
    resampled_features, resampled_target, test_size=0.2, random_state=42, stratify=resampled_target
)

# Hyperparameter optimization using Optuna
def objective(trial):
    # hyperparameters for XGBoost
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.2),
        'max_depth': trial.suggest_int('max_depth', 5, 15),  # Increased range for better flexibility
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 0.001, 10),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 0.001, 10),
        'objective': 'multi:softmax',  # Multi-class classification objective
        'num_class': 3,  # Number of classes
        'random_state': 42
    }

    # XGBoost model with hyperparameters from Optuna
    model = XGBClassifier(**params)

    # Use Stratified K-Fold Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(model, train_features, train_target, cv=cv, scoring='accuracy').mean()
    return score

# Optimize hyperparameters with Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Best hyperparameters
best_params = study.best_params
best_params['objective'] = 'multi:softmax'
best_params['num_class'] = 3
best_params['random_state'] = 42

# Train final model using the best hyperparameters
final_model = XGBClassifier(**best_params)
final_model.fit(train_features, train_target)

# Evaluate on the test set
test_predictions = final_model.predict(test_features)
test_accuracy = accuracy_score(test_target, test_predictions)

# Displaying the results
print(f"Test Accuracy of the XGBoost Model: {test_accuracy:.4f}")
print("Classification Report:")
print(classification_report(test_target, test_predictions, target_names=encoder.classes_))


# In[97]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report

# Feature Engineering
data['PulsePressure'] = data['SystolicBP'] - data['DiastolicBP']
data['MAP'] = data['DiastolicBP'] + (1/3) * (data['SystolicBP'] - data['DiastolicBP'])

# Encode Target Variable
le = LabelEncoder()
data['RiskLevel'] = le.fit_transform(data['RiskLevel'])

# Define Features & Target
X = data.drop('RiskLevel', axis=1)
y = data['RiskLevel']

# Train-Test Split BEFORE applying SMOTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle Class Imbalance ONLY on the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Introduce even more noise to the features to confuse the model
noise_factor = 0.3  # Increased noise further
X_train_noisy = X_train_resampled + noise_factor * np.random.normal(size=X_train_resampled.shape)
X_test_noisy = X_test_scaled + noise_factor * np.random.normal(size=X_test_scaled.shape)

# Define Individual Models with increased complexity
rf = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)  # Increased number of estimators and depth
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=7, random_state=42)  # Reduced learning rate for more complexity
xgb = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=7, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
lgbm = LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=7, random_state=42)
catboost = CatBoostClassifier(iterations=100, learning_rate=0.05, depth=7, verbose=0, random_state=42)
svm = SVC(probability=True, kernel='rbf', C=0.5, gamma='scale', random_state=42)  # Reduced C value for more regularization

# Voting Classifier
voting_clf = VotingClassifier(
    estimators=[ 
        ('rf', rf),
        ('gb', gb),
        ('xgb', xgb),
        ('lgbm', lgbm),
        ('catboost', catboost),
        ('svm', svm)
    ],
    voting='soft' 
)

# Train the Voting Classifier with noisy data and increased complexity models
voting_clf.fit(X_train_noisy, y_train_resampled)

# Predictions
y_pred = voting_clf.predict(X_test_noisy)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Voting Classifier Test Accuracy with Increased Noise and Complexity: {accuracy:.4f}")

# Print classification report
print(classification_report(y_test, y_pred))




