import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib

def train_model():
    """
    Trains a Decision Tree Classifier model and saves it.
    """
    # Load the training data
    training = pd.read_csv('Data/Training.csv')
    cols = training.columns[:-1]
    x = training[cols]
    y = training['prognosis']

    # Encode the target variable
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y_encoded = le.transform(y)

    # Split the data for validation (optional, but good practice)
    x_train, _, y_train, _ = train_test_split(x, y_encoded, test_size=0.33, random_state=42)

    # Train the Decision Tree Classifier
    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train)

    # Save the trained model and the label encoder
    joblib.dump(clf, 'saved_model/decision_tree.joblib')
    joblib.dump(le, 'saved_model/label_encoder.joblib')
    joblib.dump(cols, 'saved_model/training_cols.joblib')

if __name__ == '__main__':
    print("Training the model...")
    train_model()
    print("Model training complete and model saved.")
