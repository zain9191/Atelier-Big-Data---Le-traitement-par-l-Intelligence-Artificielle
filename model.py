import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

def train_model():
    print("Loading data...")
    try:
        training = pd.read_csv('Data/Training.csv')
    except FileNotFoundError:
        print("Error: 'Data/Training.csv' not found. Please ensure the file is in the Data folder.")
        return

    cols = training.columns[:-1]
    x = training[cols]
    y = training['prognosis']

    le = preprocessing.LabelEncoder()
    le.fit(y)
    y_encoded = le.transform(y)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y_encoded,
        test_size=0.33,
        random_state=42,
        stratify=y_encoded
    )

    print("Training Decision Tree...")
    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {test_accuracy:.4f}")

    if not os.path.exists('saved_model'):
        os.makedirs('saved_model')

    print("Saving models...")
    joblib.dump(clf, 'saved_model/decision_tree.joblib')
    joblib.dump(le, 'saved_model/label_encoder.joblib')
    joblib.dump(cols, 'saved_model/training_cols.joblib')
    print("Model trained successfully!")

if __name__ == '__main__':
    train_model()
