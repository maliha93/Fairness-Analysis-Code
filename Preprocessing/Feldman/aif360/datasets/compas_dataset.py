import os

import pandas as pd

from aif360.datasets import StandardDataset


default_mappings = {
    'label_maps': [{0.0: 'Did recid.', 1.0: 'No recid.'}],
    'protected_attribute_maps': [{1.0: 'Other', 0.0: 'African-American'}]
}

def default_preprocessing(df):
    return df[(df.two_year_recid != -1)]

class CompasDataset(StandardDataset):
    """ProPublica COMPAS Dataset.

    See :file:`aif360/data/raw/compas/README.md`.
    """

    def __init__(self, label_name='two_year_recid', favorable_classes=[1],
                 protected_attribute_names=['Race'],
                 privileged_classes=[['Caucasian', 'Hispanic', 'Asian', 'Native American', 'Other']],
                 instance_weights_name=None,
                 categorical_features=['Sex'],
                 features_to_keep=['Sex', 'Age', 'Race', 'Prior'],
                 features_to_drop=[], na_values=[],
                 custom_preprocessing=default_preprocessing,
                 metadata=default_mappings, fname=None):
       
        if fname == None:
            filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                '..', 'data', 'compas.csv')
        else:
            filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                '..', fname)
        

        try:
            df = pd.read_csv(filepath, na_values=na_values)
            #df['Sex'] = df['Sex'].replace({'Female': 0.0, 'Male': 1.0})
        except IOError as err:
            print("IOError: {}".format(err))
            print("To use this class, please download the following file:")
            print("\n\thttps://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv")
            print("\nand place it, as-is, in the folder:")
            print("\n\t{}\n".format(os.path.abspath(os.path.join(
               os.path.abspath(__file__), '..', '..', 'data', 'raw', 'compas'))))
            import sys
            sys.exit(1)

        super(CompasDataset, self).__init__(df=df, label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop, na_values=na_values,
            custom_preprocessing=custom_preprocessing, metadata=metadata)
