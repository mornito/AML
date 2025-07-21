# 🌸 Iris Flower Classifier: Your First Machine Learning Model

# Welcome! In this notebook, you'll build a simple machine learning (ML) model to classify Iris flowers
# based on their features like petal and sepal size.

# What is Machine Learning?
# Machine learning is like teaching a computer by showing it examples.
# Just like you learn by looking at different types of flowers, the computer learns from data.

# This guide is made for absolute beginners—even if you're in 10th grade and new to coding!

# 📚 Step 1: Import libraries
# These are like your tools. We use them to load data, train the model, and check results.

from sklearn.datasets import load_iris  # Built-in flower data
from sklearn.model_selection import train_test_split  # Split data into training and testing
from sklearn.ensemble import RandomForestClassifier  # A type of model
from sklearn.metrics import accuracy_score, classification_report  # To measure how good the model is
import pandas as pd  # To work with data like a table
import matplotlib.pyplot as plt  # To make charts

# If running in Jupyter/Colab, uncomment the next line:
# %matplotlib inline

# 🌼 Step 2: Load the Iris dataset
# The dataset has 150 rows. Each row is a flower with features like petal length.

iris = load_iris()
X = iris.data  # Features (numbers we use to make predictions)
y = iris.target  # Labels (flower types: 0 = Setosa, 1 = Versicolor, 2 = Virginica)
feature_names = iris.feature_names
target_names = iris.target_names

# Let's turn it into a table for better understanding
df = pd.DataFrame(X, columns=feature_names)
df['species'] = [target_names[i] for i in y]
print("First 5 rows of the dataset:")
print(df.head())

# 📊 Step 3: Visualize the data
# Let's make a scatter plot to see how the features look.
# Each dot is a flower. Colors show different species.

pd.plotting.scatter_matrix(df[feature_names], figsize=(10, 8), c=y, cmap='viridis', diagonal='hist')
plt.suptitle('Iris Feature Scatter Matrix', fontsize=16)
plt.show()

# ✂️ Step 4: Split the data into training and test sets
# We train the model on 80% of data, and test it on 20% to see how it performs on new flowers.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {len(X_train)}\nTest samples: {len(X_test)}")

# 🤖 Step 5: Train a Random Forest model
# A model is like a smart rulebook the computer writes after looking at the examples.

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
print("✅ Model trained!")

# 🧪 Step 6: Test the model
# We check if the model guesses the right species for the test data.

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy:.2%}\n")
print("📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# 🎉 Awesome! You just trained your first machine learning model.
# Now try:
# - Changing the model type (like KNeighborsClassifier)
# - Using only 2 features and checking the accuracy
# - Asking a question: Can I make it better?

print("You’re officially an ML explorer now! 🚀")

