import pandas as pd

source_file = '../data/adult/dataset-data+test.csv'
binarized_file = '../data/adult.csv'


def Mapping(tuple):
    # age, 37
    tuple['age'] = 1 if tuple['age'] > 37 else 0
    # workclass
    tuple['workclass'] = 'NonPrivate' if tuple['workclass'] != 'Private' else 'Private'
    # edunum
    tuple['edu_level'] = 1 if tuple['edu_level'] > 9 else 0
    # maritial statue
    tuple['marital_status'] = "Marriedcivspouse" if tuple['marital_status'] == "Married-civ-spouse" else "nonMarriedcivspouse"
    # occupation
    tuple['occupation'] = "Craftrepair" if tuple['occupation'] == "Craft-repair" else "NonCraftrepair"
    # relationship
    tuple['relationship'] = "NotInFamily" if tuple['relationship'] == "Not-in-family" else "InFamily"
    # race
    tuple['race'] = 'NonWhite' if tuple['race'] != "White" else 'While'
    # hours per week
    tuple['hours_per_week'] = 1 if tuple['hours_per_week'] > 40 else 0
    # native country
    tuple['native_country'] = "US" if tuple['native_country'] == "United-States" else "NonUS"
    return tuple


def Binarize():
    data = pd.read_csv(source_file)
    data = data.drop(['fnlwgt', 'education', 'capital-gain', 'capital-loss'], axis=1)
    data = data.apply(Mapping, axis=1)
    data.to_csv(binarized_file, index=False)


if __name__ == "__main__":
    Binarize()
