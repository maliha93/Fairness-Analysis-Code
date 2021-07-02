from aif360.datasets import AdultDataset, GermanDataset, CompasDataset, CreditDataset
import pandas as pd
import numpy as np

def load_preproc_data_adult(protected_attributes=None, sub_samp=False, balance=False, fname=None):
    def custom_preprocessing(df):
        """The custom pre-processing function is adapted from
            https://github.com/fair-preprocessing/nips2017/blob/master/Adult/code/Generate_Adult_Data.ipynb
            If sub_samp != False, then return smaller version of dataset truncated to tiny_test data points.
        """

        
        def group_race(x):
            if x == "White":
                return 1.0
            else:
                return 0.0
        def native_country(x):
            if x == "United-States":
                return "US"
            else:
                return "Non-US"
        
        # Recode sex, race, native country
        df['sex'] = df['sex'].replace({'Female': 0.0, 'Male': 1.0})
        df['race'] = df['race'].apply(lambda x: group_race(x))
        df['native_country'] = df['native_country'].apply(lambda x: native_country(x))

        if sub_samp and not balance:
            df = df.sample(sub_samp)
        if sub_samp and balance:
            df_0 = df[df['Income Binary'] == '<=50K']
            df_1 = df[df['Income Binary'] == '>50K']
            df_0 = df_0.sample(int(sub_samp/2))
            df_1 = df_1.sample(int(sub_samp/2))
            df = pd.concat([df_0, df_1])
        return df

    XD_features = ['age', 'workclass', 'edu_level', 'marital_status', 'occupation', 'relationship',\
    'race', 'sex', 'hours_per_week', 'native_country' ]
    D_features = ['sex', 'race'] if protected_attributes is None else protected_attributes
    Y_features = ['income']
    X_features = list(set(XD_features)-set(D_features))
    categorical_features = ['workclass', 'marital_status', 'native_country',\
    'relationship', 'occupation']

    # privileged classes
    all_privileged_classes = {"sex": [1.0],
                              "race": [1.0]}

    # protected attribute maps
    all_protected_attribute_maps = {"sex": {1.0: 'Male', 0.0: 'Female'},
                                    "race": {1.0: 'White', 0.0: 'Non-white'}}

    return AdultDataset(
        label_name=Y_features[0],
        favorable_classes=[1],
        protected_attribute_names=D_features,
        privileged_classes=[all_privileged_classes[x] for x in D_features],
        instance_weights_name=None,
        categorical_features=categorical_features,
        features_to_keep=X_features+Y_features+D_features,
        na_values=['?'],
        metadata={'label_maps': [{1.0: '>50K', 0.0: '<=50K'}],
                  'protected_attribute_maps': [all_protected_attribute_maps[x]
                                for x in D_features]},
        custom_preprocessing=custom_preprocessing, fname=fname)

def load_preproc_data_compas(protected_attributes=None, fname=None):
    def custom_preprocessing(df):
        """The custom pre-processing function is adapted from
            https://github.com/fair-preprocessing/nips2017/blob/master/compas/code/Generate_Compas_Data.ipynb
        """

        df = df[['Sex', 'Age', 'Race', 'Prior', 'two_year_recid']]

        # Indices of data samples to keep
        #ix = (df['two_year_recid'] != -1)
        #df = df.loc[ix,:]
        
        
        # Restrict the features to use
        dfcutQ = df.copy()

        
        # Quantize length of stay
        def adjustAge(x):
            if x == '25 - 45':
                return '25 to 45'
            else:
                return x

        def group_race(x):
            if x in ['Caucasian', 'Native American', 'Hispanic', 'Asian', 'Other']:
                return 1.0
            else:
                return 0.0
        
        def adjust_sex(x):
            if x in [1.0]:
                return 'Male'
            else:
                return 'Female'

        #dfcutQ['age_cat'] = dfcutQ['age_cat'].apply(lambda x: adjustAge(x))

        # Recode sex and race
        dfcutQ['Sex'] = dfcutQ['Sex'].apply(lambda x: adjust_sex(x))
        dfcutQ['Race'] = dfcutQ['Race'].apply(lambda x: group_race(x))

        features = ['Sex', 'Age', 'Race', 'Prior', 'two_year_recid']

        # Pass vallue to df
        df = dfcutQ[features]

        return df

    XD_features = ['Sex', 'Age', 'Race', 'Prior']
    D_features = ['Race']  if protected_attributes is None else protected_attributes
    Y_features = ['two_year_recid']
    X_features = list(set(XD_features)-set(D_features))
    categorical_features = ['Sex']

    # privileged classes
    all_privileged_classes = {"Race": [1.0]}

    # protected attribute maps
    all_protected_attribute_maps = {"Race": {1.0: 'Other', 0.0: 'African-American'}}


    return CompasDataset(
        label_name=Y_features[0],
        favorable_classes=[1],
        protected_attribute_names=D_features,
        privileged_classes=[all_privileged_classes[x] for x in D_features],
        instance_weights_name=None,
        categorical_features=categorical_features,
        features_to_keep=X_features+Y_features+D_features,
        na_values=[],
        metadata={'label_maps': [{0.0: 'Did recid.', 1.0: 'No recid.'}],
                  'protected_attribute_maps': [all_protected_attribute_maps[x]
                                for x in D_features]},
        custom_preprocessing=custom_preprocessing, fname=fname)

def load_preproc_data_german(protected_attributes=None, fname=None):
    """
    Load and pre-process german credit dataset.
    Args:
        protected_attributes(list or None): If None use all possible protected
            attributes, else subset the protected attributes to the list.

    Returns:
        GermanDataset: An instance of GermanDataset with required pre-processing.

    """
    def custom_preprocessing(df):
        """ Custom pre-processing for German Credit Data
        """

        def group_credit_hist(x):
            if x in ['A30', 'A31', 'A32']:
                return 'None/Paid'
            elif x == 'A33':
                return 'Delay'
            elif x == 'A34':
                return 'Other'
            else:
                return 'NA'

        def group_employ(x):
            if x == 'A71':
                return 'Unemployed'
            elif x in ['A72', 'A73']:
                return '1-4 years'
            elif x in ['A74', 'A75']:
                return '4+ years'
            else:
                return 'NA'

        def group_savings(x):
            if x in ['A61', 'A62']:
                return '<500'
            elif x in ['A63', 'A64']:
                return '500+'
            elif x == 'A65':
                return 'Unknown/None'
            else:
                return 'NA'

        def group_status(x):
            if x in ['A11', 'A12']:
                return '<200'
            elif x in ['A13']:
                return '200+'
            elif x == 'A14':
                return 'None'
            else:
                return 'NA'

        df['Sex'] = df['Sex'].replace({'Female': 0.0, 'Male': 1.0})

        # group credit history, savings, and status
        df['Credit_history'] = df['Credit_history'].apply(lambda x: group_credit_hist(x))
        df['Savings'] = df['Savings'].apply(lambda x: group_savings(x))
        #df['employment'] = df['employment'].apply(lambda x: group_employ(x))
        #df['age'] = df['age'].apply(lambda x: np.float(x >= 26))
        df['Status'] = df['Status'].apply(lambda x: group_status(x))
        return df

    # Feature partitions
    XD_features = ['Month', 'Credit_amount', 'Investment', 'Age', 'Sex', 'Status', 'Credit_history',\
                   'Savings', 'Property', 'Housing']
    D_features = ['Sex'] if protected_attributes is None else protected_attributes
    Y_features = ['credit']
    X_features = list(set(XD_features)-set(D_features))
    categorical_features = ['Credit_history', 'Status', 'Savings', 'Property', 'Housing']

    # privileged classes
    all_privileged_classes = {"Sex": [1.0]}

    # protected attribute maps
    all_protected_attribute_maps = {"Sex": {1.0: 'Male', 0.0: 'Female'}}

    return GermanDataset(
        label_name=Y_features[0],
        favorable_classes=[1],
        protected_attribute_names=D_features,
        privileged_classes=[all_privileged_classes[x] for x in D_features],
        instance_weights_name=None,
        categorical_features=categorical_features,
        features_to_keep=X_features+Y_features+D_features,
        metadata={ 'label_maps': [{1.0: 'Good Credit', 0.0: 'Bad Credit'}],
                   'protected_attribute_maps': [all_protected_attribute_maps[x]
                                for x in D_features]},
        custom_preprocessing=custom_preprocessing, fname=fname)
