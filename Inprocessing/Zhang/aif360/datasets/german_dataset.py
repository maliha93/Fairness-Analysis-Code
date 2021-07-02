import os

import pandas as pd

from aif360.datasets import StandardDataset


default_mappings = {
    'label_maps': [{1.0: 'Good Credit', 0.0: 'Bad Credit'}],
    'protected_attribute_maps': [{1.0: 'Male', 0.0: 'Female'}]
}


class GermanDataset(StandardDataset):
    """German credit Dataset.

    See :file:`aif360/data/raw/german/README.md`.
    """

    def __init__(self, label_name='credit', favorable_classes=[1],
                 protected_attribute_names=['Sex'],
                 privileged_classes=[['Male']],
                 instance_weights_name=None,
                 categorical_features=[ 'Status', 'Credit_history',\
                   'Savings', 'Property', 'Housing'],
                 features_to_keep=[], features_to_drop=[],
                 na_values=[], custom_preprocessing=None,
                 metadata=default_mappings, fname=None):
        if fname == None:
            filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'data', 'german.csv')
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
