import os

import pandas as pd

from aif360.datasets import StandardDataset


default_mappings = {
    'label_maps': [{1.0: 'Good Credit', 0.0: 'Bad Credit'}],
    'protected_attribute_maps': [{1.0: 'Male', 0.0: 'Female'}]
}

def custom_preprocessing_german(df):
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

        #df['Sex'] = df['Sex'].replace({'Female': 0.0, 'Male': 1.0})

        # group credit history, savings, and status
        df['Credit_history'] = df['Credit_history'].apply(lambda x: group_credit_hist(x))
        df['Savings'] = df['Savings'].apply(lambda x: group_savings(x))
        #df['employment'] = df['employment'].apply(lambda x: group_employ(x))
        #df['age'] = df['age'].apply(lambda x: np.float(x >= 26))
        df['Status'] = df['Status'].apply(lambda x: group_status(x))
        return df


class GermanDataset(StandardDataset):
    """German credit Dataset.

    See :file:`aif360/data/raw/german/README.md`.
    """

    def __init__(self, label_name='credit', favorable_classes=[1],
                 protected_attribute_names=['Sex'],
                 privileged_classes=[['Male']],
                 instance_weights_name=None,
                 categorical_features=[ 'Age', 'Month', 'Investment', 'Credit_amount', 'Status', 'Credit_history',\
                   'Savings', 'Property', 'Housing'],
                 features_to_keep=[], features_to_drop=[],
                 na_values=[], custom_preprocessing=custom_preprocessing_german,
                 metadata=default_mappings, fname=None):
        
        if fname == None:
            filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'data', 'german_bin.csv')
        else:
            filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', fname)
            
        try:
            df = pd.read_csv(filepath, na_values=na_values)
        except IOError as err:
            print("IOError: {}".format(err))
            print("To use this class, please download the following files:")
            print("\n\thttps://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data")
            print("\thttps://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.doc")
            print("\nand place them, as-is, in the folder:")
            print("\n\t{}\n".format(os.path.abspath(os.path.join(
                os.path.abspath(__file__), '..', '..', 'data', 'raw', 'german'))))
            import sys
            sys.exit(1)

        super(GermanDataset, self).__init__(df=df, label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop, na_values=na_values,
            custom_preprocessing=custom_preprocessing, metadata=metadata)
