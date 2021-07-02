import os

import pandas as pd

from aif360.datasets import StandardDataset


default_mappings = {
    'label_maps': [{1.0: 1.0, 0.0: 0.0}],
    'protected_attribute_maps': [{1.0: 'Male', 0.0: 'Female'}]
}

class AdultDataset(StandardDataset):
    """Adult Census Income Dataset.

    See :file:`aif360/data/raw/adult/README.md`.
    """

    def __init__(self, label_name='income',
                 favorable_classes=[1],
                 protected_attribute_names=['sex'],
                 privileged_classes=[['Male']],
                 instance_weights_name=None,
                 categorical_features=[ 'race', 'workclass',
                     'marital_status', 'occupation', 'relationship',
                     'native_country'],
                 features_to_keep=[], features_to_drop=[],
                 na_values=['?'], custom_preprocessing=None,
                 metadata=default_mappings, fname=None):
        
        if fname == None:
            train_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  '..', 'data', 'adult.csv')
        else:
            filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', fname)
        
        try:
            df = pd.read_csv(train_path, na_values=na_values)
            
        except IOError as err:
            print("IOError: {}".format(err))
            print("To use this class, please download the following files:")
            print("\n\thttps://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data")
            print("\thttps://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test")
            print("\thttps://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names")
            print("\nand place them, as-is, in the folder:")
            print("\n\t{}\n".format(os.path.abspath(os.path.join(
               os.path.abspath(__file__), '..', '..', 'data', 'raw', 'adult'))))
            import sys
            sys.exit(1)

        #df = pd.concat([train, test], ignore_index=True)
        #df = df.sample(frac=1).reset_index(drop=True)

        super(AdultDataset, self).__init__(df=df, label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop, na_values=na_values,
            custom_preprocessing=custom_preprocessing, metadata=metadata)
