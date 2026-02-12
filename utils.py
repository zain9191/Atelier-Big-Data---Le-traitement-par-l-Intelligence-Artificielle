import csv

def getSeverityDict():
    severityDictionary = dict()
    with open('MasterData/symptom_severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) > 1 and row[1].isdigit():
                _diction={row[0]:int(row[1])}
                severityDictionary.update(_diction)
    return severityDictionary

def getDescription():
    description_list = dict()
    with open('MasterData/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) > 1:
                _description={row[0]:row[1]}
                description_list.update(_description)
    return description_list

def getprecautionDict():
    precautionDictionary = dict()
    with open('MasterData/symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) > 4:
                _prec={row[0]:[row[1],row[2],row[3],row[4]]}
                precautionDictionary.update(_prec)
    return precautionDictionary

def getspecialistDict():
    specialistDictionary = dict()
    with open('MasterData/symptom_specialist.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None)  # Skip header row
        for row in csv_reader:
            if len(row) > 1:
                specialistDictionary[row[0]] = row[1]
    return specialistDictionary
