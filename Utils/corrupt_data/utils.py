import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def bin_adult_data(df):
    
    def age_bin(x):
        if (x < 25):
            return "less_than_25"
        elif (x >=25 and x <=45):
            return "25_to_45"
        else:
            return "greater_than_45"
        
    def education_bin(x):
        if (x < 6):
            return "less_than_6"
        elif (x >=6 and x <=12):
            return "6_to_12"
        else:
            return "greater_than_12"
        
    def hours_bin(x):
        if (x < 25):
            return "less_than_25"
        elif (x >=25 and x <=40):
            return "25_to_40"
        else:
            return "greater_than_40"
        
    def country_bin(x):
        if (x == 'United-States'):
            return "United-States"
        else:
            return "Non-US"
        
    df['native_country'] = df['native_country'].apply(lambda x: country_bin(x))
    df['age'] = df['age'].apply(lambda x: age_bin(x))
    df['edu_level'] = df['edu_level'].apply(lambda x: education_bin(x))
    df['hours_per_week'] = df['hours_per_week'].apply(lambda x: hours_bin(x))
    return df

def bin_compas_data(df):
    
    def age_bin(x):
        if (x < 25):
            return "less_than_25"
        elif (x >=25 and x <=45):
            return "25_to_45"
        else:
            return "greater_than_45"
        
    def prior_bin(x):
        if (x < 3):
            return "less_than_3"
        elif (x >=3 and x <=10):
            return "3_to_10"
        else:
            return "greater_than_10"
        
    def race_bin(x):
        if (x == 'African-American'):
            return "African-American"
        else:
            return "Other"
        
    df['Age'] = df['Age'].apply(lambda x: age_bin(x))
    df['Prior'] = df['Prior'].apply(lambda x: prior_bin(x))
    df['Race'] = df['Race'].apply(lambda x: race_bin(x))
    return df
    
def bin_german_data(df):
    
    def age_bin(x):
        if (x < 25):
            return "less_than_25"
        elif (x >=25 and x <=45):
            return "25_to_45"
        else:
            return "greater_than_45"
        
    def month_bin(x):
        if (x < 20):
            return "less_than_20"
        elif (x >=20 and x <=40):
            return "20_to_40"
        else:
            return "greater_than_40"
    
    def credit_bin(x):
        if (x < 5000):
            return "low"
        elif (x >=5000 and x <=10000):
            return "medium"
        else:
            return "high"
        
    def invest_bin(x):
        if (x < 2):
            return "less_than_2"
        elif (x >=2 and x <=3):
            return "2_to_3"
        else:
            return "greater_than_3"
        
    df['Age'] = df['Age'].apply(lambda x: age_bin(x))
    df['Month'] = df['Month'].apply(lambda x: month_bin(x))
    df['Credit_amount'] = df['Credit_amount'].apply(lambda x: credit_bin(x))
    df['Investment'] = df['Investment'].apply(lambda x: invest_bin(x))
    return df 

def binarize_adult_data(df):
    
    def age_bin(x):
        if (x < 30):
            return "under_30"
        else:
            return "above_30"
        
    def education_bin(x):
        if (x < 12):
            return "under_12"
        else:
            return "above_12"
        
    def hours_bin(x):
        if (x < 25):
            return "under_25"
        else:
            return "above_25"
        
    def country_bin(x):
        if (x == 'United-States'):
            return "United-States"
        else:
            return "Non-US"
        
    def workclass_bin(x):
        if x in ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Without-pay' ]:
            return "Private"
        else:
            return "Non-Private"
        
    def marital_bin(x):
        if x in ['Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse']:
            return "Married"
        else:
            return "Non-married"
        
    def relationship_bin(x):
        if x in ['Own-child', 'Husband', 'Wife', 'Other-relative']:
            return "Family"
        else:
            return "Not-in-family"
    
    def occupation_bin(x): 
        if x in ['Machine-op-inspct', 'Prof-specialty','Exec-managerial', 'Tech-support']:
            return "Technical"
        else:
            return "Not-technical"
        
        
    df['native_country'] = df['native_country'].apply(lambda x: country_bin(x))
    df['age'] = df['age'].apply(lambda x: age_bin(x))
    df['edu_level'] = df['edu_level'].apply(lambda x: education_bin(x))
    df['workclass'] = df['workclass'].apply(lambda x: workclass_bin(x))
    df['marital_status'] = df['marital_status'].apply(lambda x: marital_bin(x))
    df['relationship'] = df['relationship'].apply(lambda x: relationship_bin(x))
    df['occupation'] = df['occupation'].apply(lambda x: occupation_bin(x))
    df['hours_per_week'] = df['hours_per_week'].apply(lambda x: hours_bin(x))
    
    return df

def binarize_compas_data(df):
    
    def age_bin(x):
        if (x < 25):
            return "under_25"
        else:
            return "above_25"
        
    def prior_bin(x):
        if (x < 5):
            return "under_5"
        else:
            return "over_5"
        
    def race_bin(x):
        if (x == 'African-American'):
            return "African-American"
        else:
            return "Other"
        
    df['Age'] = df['Age'].apply(lambda x: age_bin(x))
    df['Prior'] = df['Prior'].apply(lambda x: prior_bin(x))
    df['Race'] = df['Race'].apply(lambda x: race_bin(x))
    return df
    
def binarize_german_data(df):
    
    def age_bin(x):
        if (x < 25):
            return "under_25"
        else:
            return "above_45"
        
    def month_bin(x):
        if (x < 40):
            return "under_40"
        else:
            return "above_40"
    
    def credit_bin(x):
        if x <=10000:
            return "average"
        else:
            return "high"
        
    def invest_bin(x):
        if (x < 3):
            return "under_3"
        else:
            return "above_3"
        
    def status_bin(x):
        if x in ['A11', 'A12', 'A13']:
            return "Exists"
        else:
            return "Not-exists"
        
    def history_bin(x):
        if x in ['A30', 'A31', 'A32']:
            return "Duly_paid"
        else:
            return "Delayed"
        
    def savings_bin(x):
        if x in ['A61', 'A63', 'A64', 'A62']:
            return "Yes"
        else:
            return "No"
    
    def property_bin(x):
        if x in ['A121', 'A122', 'A123']:
            return "Yes"
        else:
            return "No"
        
    def housing_bin(x):
        if x in ['A152', 'A153']:
            return "Not-rented"
        else:
            return "Rented"
        
    df['Age'] = df['Age'].apply(lambda x: age_bin(x))
    df['Month'] = df['Month'].apply(lambda x: month_bin(x))
    df['Credit_amount'] = df['Credit_amount'].apply(lambda x: credit_bin(x))
    df['Investment'] = df['Investment'].apply(lambda x: invest_bin(x))
    df['Status'] = df['Status'].apply(lambda x: status_bin(x))
    df['Credit_history'] = df['Credit_history'].apply(lambda x: history_bin(x))
    df['Savings'] = df['Savings'].apply(lambda x: savings_bin(x))
    df['Property'] = df['Property'].apply(lambda x: property_bin(x))
    df['Housing'] = df['Housing'].apply(lambda x: housing_bin(x))
    return df 