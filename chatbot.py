import re
import numpy as np
import joblib
import pandas as pd
from sklearn.tree import _tree
from utils import getDescription, getSeverityDict, getprecautionDict, getspecialistDict

# Load all the necessary artifacts
clf = joblib.load('saved_model/decision_tree.joblib')
le = joblib.load('saved_model/label_encoder.joblib')
training_cols = joblib.load('saved_model/training_cols.joblib')
description_list = getDescription()
severityDictionary = getSeverityDict()
precautionDictionary = getprecautionDict()
specialistDictionary = getspecialistDict()

symptoms_dict = {symptom: index for index, symptom in enumerate(training_cols)}
training = pd.read_csv('Data/Training.csv')
reduced_data = training.groupby(training['prognosis']).max()

def normalize_symptom(symptom):
    return symptom.strip().lower().replace(" ", "_")


def check_pattern(dis_list, inp):
    if not inp:
        return 0, []

    normalized = normalize_symptom(inp)
    regexp = re.compile(re.escape(normalized))
    pred_list = [item for item in dis_list if regexp.search(item.lower())]
    return (1, pred_list) if pred_list else (0, [])

def sec_predict(symptoms_exp):
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        normalized_item = normalize_symptom(item)
        if normalized_item in symptoms_dict:
            input_vector[symptoms_dict[normalized_item]] = 1

    input_df = pd.DataFrame([input_vector], columns=training_cols)
    predicted_encoded = clf.predict(input_df)
    return le.inverse_transform(predicted_encoded)[0]

def print_disease(node):
    node = node[0]
    val = node.nonzero()
    disease = le.inverse_transform(val[0])
    return list(map(lambda x: x.strip(), list(disease)))

def calc_condition(exp, days):
    sum = 0
    for item in exp:
        if item in severityDictionary:
            sum = sum + severityDictionary[item]
    if (sum * days) / (len(exp) + 1) > 13:
        return "You should take the consultation from a doctor."
    else:
        return "It might not be that bad, but you should take precautions."

def predict_initial(symptom):
    symptom = normalize_symptom(symptom)
    if symptom not in symptoms_dict:
        raise ValueError("Unknown symptom. Please choose one from the suggestions.")

    tree_ = clf.tree_
    feature_name = [
        training_cols[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    
    symptoms_present = []

    def recurse(node, depth):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == symptom:
                val = 1
            else:
                val = 0
            if val <= threshold:
                return recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                return recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            red_cols = reduced_data.columns
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            return present_disease, list(symptoms_given)

    return recurse(0, 1)


def predict_final(initial_disease, symptoms_exp, days):
    second_prediction = sec_predict(symptoms_exp)

    severity_message = calc_condition(symptoms_exp, days)

    text = ""
    if initial_disease == second_prediction:
        text = f"You may have {initial_disease}."
    else:
        text = f"You may have {initial_disease} or {second_prediction}."

    description_present = description_list.get(initial_disease, "")
    description_second = description_list.get(second_prediction, "")
    precautions = precautionDictionary.get(initial_disease, [])
    specialist_present = specialistDictionary.get(initial_disease, "General Physician")
    specialist_second = specialistDictionary.get(second_prediction, "General Physician")

    return (
        text,
        description_present,
        description_second,
        precautions,
        severity_message,
        specialist_present,
        specialist_second,
    )


if __name__ == '__main__':
    # This is for testing the chatbot logic from the command line.
    print("HealthCare ChatBot")
    symptom = input("Enter the symptom you are experiencing: ")
    
    disease, related_symptoms = predict_initial(symptom)
    print("Predicted disease:", disease)
    print("Related symptoms:", related_symptoms)
    
    days = int(input("Okay. From how many days? : "))
    
    text, desc1, desc2, precautions, severity, specialist1, specialist2 = predict_final(disease[0], [symptom], days)
    
    print(text)
    print(desc1)
    if desc2 and desc1 != desc2:
        print(desc2)
    print(precautions)
    print(severity)
    print("Recommended specialist:", specialist1)
    if specialist1 != specialist2:
        print("Alternative specialist:", specialist2)
