# Step 1 => Importing required libraries
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import accuracy_score  
from sklearn.preprocessing import StandardScaler  

# Step 2 => Define function to load dataset
def load_dataset(file_path):
    data = pd.read_csv(file_path)
    print(data.info())  
    return data

# Step 3 => Define function to check for missing values
def check_missing_values(data):
    return data.isnull().sum()

# Step 4 => Define function to count fraud and legit transactions
def count_classes(data):
    return data['Class'].value_counts()

# Step 5 => Define function to separate legit and fraud transactions
def separate_transactions(data):
    legit_transactions = data[data.Class == 0]
    fraud_transactions = data[data.Class == 1]
    return legit_transactions, fraud_transactions

# Step 6 => Define function to display statistical details
def show_statistics(legit_transactions, fraud_transactions):
    print(legit_transactions.Amount.describe())
    print(fraud_transactions.Amount.describe())

# Step 7 => Define function to get mean of each column grouped by Class
def get_class_mean(data):
    return data.groupby('Class').mean()

# Step 8 => Define function to balance dataset
def balance_dataset(legit_transactions, fraud_transactions, sample_size=492):
    legit_sample = legit_transactions.sample(n=sample_size)
    balanced_data = pd.concat([legit_sample, fraud_transactions], axis=0)
    return balanced_data

# Step 9 => Define function to separate features and target
def separate_features_target(data):
    features = data.drop(columns='Class', axis=1)
    target = data['Class']
    return features, target

# Step 10 => Define function to normalize features
def normalize_features(features):
    scaler = StandardScaler()
    norm_features = scaler.fit_transform(features)
    return norm_features

# Step 11 => Define function to split dataset
def split_dataset(features, target):
    return train_test_split(features, target, test_size=0.2, stratify=target, random_state=2)

# Step 12 => Define function to train logistic regression model
def train_model(X_train, Y_train):
    model = LogisticRegression(solver='liblinear', C=1.0, max_iter=500)
    model.fit(X_train, Y_train)
    return model

# Step 13 => Define function to evaluate model
def evaluate_model(model, X_train, Y_train, X_test, Y_test):
    train_predictions = model.predict(X_train)
    training_accuracy = accuracy_score(train_predictions, Y_train)
    test_predictions = model.predict(X_test)
    test_accuracy = accuracy_score(test_predictions, Y_test)
    return training_accuracy, test_accuracy

# Step 14 => Define function to visualize results
def visualize_results(training_accuracy, test_accuracy):
    labels = ['Training Accuracy', 'Test Accuracy']
    sizes = [training_accuracy * 100, test_accuracy * 100]
    colors = ['skyblue', 'lightcoral']
    explode = (0.1, 0)
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, explode=explode, shadow=True)
    plt.title('Model Accuracy Comparison')
    plt.show()

# Step 15 => Running the defined functions
data = load_dataset('.\\Dataset\\creditcard.csv')
print(check_missing_values(data))
print(count_classes(data))
legit_transactions, fraud_transactions = separate_transactions(data)
show_statistics(legit_transactions, fraud_transactions)
print(get_class_mean(data))
balanced_data = balance_dataset(legit_transactions, fraud_transactions)
features, target = separate_features_target(balanced_data)
norm_features = normalize_features(features)
X_train, X_test, Y_train, Y_test = split_dataset(norm_features, target)
model = train_model(X_train, Y_train)
training_accuracy, test_accuracy = evaluate_model(model, X_train, Y_train, X_test, Y_test)
print('Accuracy on Training data:', training_accuracy)
print('Accuracy score on Test Data:', test_accuracy)
visualize_results(training_accuracy, test_accuracy)
