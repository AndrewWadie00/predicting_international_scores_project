# Importing necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Looping through the directory to find the CSV file
import os
for dirname, _, filenames in os.walk('enter your path to files here'): #modify file path here
    for filename in filenames:
        print(os.path.join(dirname, filename))
        filepath= os.path.join(dirname, filename)
        rows = []
        if filename == "results.csv":
            with open(filepath, 'r') as file:
                csvreader = pd.read_csv(filepath)
                rows.append(csvreader)

# Creating DataFrame from the CSV data
df= pd.DataFrame(rows[0])

# Adding an index column
i= range(1,len(df)+1)
df["index"]= i
df.set_index("index", inplace=True)

# Extracting year, month, and day from the 'date' column
df[["year","month","day"]] = df["date"].str.split("-", expand=True)

# Determining home team's result (win, lose, or draw)
df["home_result"]= np.where(df["home_score"] > df["away_score"], "win", np.where(df["home_score"] < df["away_score"], "lose", "draw"))

# Removing unnecessary columns
columns_to_be_removed= ["month","day","home_score","away_score","city","date"]
df.drop(columns=columns_to_be_removed, inplace=True)

# Converting 'year' column to integer 
df["year"] = df["year"].astype(int)


# Splitting data into features (x) and target (y)
y= df["home_result"]
x= df[["home_team","away_team","tournament","country","neutral"]]
x_train,x_test,y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=42)

# Calculating baseline accuracy
y_baseline= ["win"]* len(y_train)
accuracy_baseline = accuracy_score(y_train, y_baseline)
print("Accuracy Baseline:", accuracy_baseline)

# Defining categorical columns for one-hot encoding
categorical_cols = ["home_team","away_team","tournament","country"]

# Creating a ColumnTransformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_cols)
    ],
    remainder='passthrough'
)

# Creating a pipeline with preprocessing and logistic regression
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Training the model using a sliding window approach
y_pred= []
window_size=4185

for i in range(1,len(y_train) - window_size + 1, window_size):
        x_window = x_train.iloc[i:i+window_size]
        y_window = y_train.iloc[i:i+window_size]
        # Fit the pipeline on the training data
        pipeline.fit(x_window, y_window)

        # Make predictions on the training data
        y_pred_window = pipeline.predict(x_window)

        y_pred.extend(y_pred_window)

# Evaluating model's performance on training data
y_train = y_train.iloc[1:len(y_pred)+1]
accuracy = accuracy_score(y_train, y_pred)
print("Training Accuracy:", accuracy)

# Testing the model using a sliding window approach
y_pred_test= []

for i in range(1, len(y_test) - window_size + 1, window_size):
        x_window_test = x_test.iloc[i:i+window_size]
        y_window_test = y_test.iloc[i:i+window_size]
        # Fit the pipeline on the testing data
        pipeline.fit(x_window_test, y_window_test)

        # Make predictions on the testing data
        y_pred_window_test = pipeline.predict(x_window_test)

        y_pred_test.extend(y_pred_window_test)

# Evaluating model's performance on testing data
y_test = y_test.iloc[1:len(y_pred_test)+1]
accuracy = accuracy_score(y_test, y_pred_test)
print("Testing Accuracy:", accuracy)

# Computing confusion matrix
cm = confusion_matrix(y_test, y_pred_test)

# Plotting confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', 
            xticklabels=['win', 'lose', 'draw'], 
            yticklabels=['win', 'lose', 'draw'])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()