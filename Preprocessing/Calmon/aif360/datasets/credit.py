import os
import pandas as pd
from aif360.datasets import StandardDataset

default_mappings = {
    'label_maps': [{1.0: 'Good Credit', 0.0: 'Bad Credit'}],
    'protected_attribute_maps': [{1.0: 'Male', 0.0: 'Female'}],
}

class CreditDataset(StandardDataset):
    """Taiwan credit Dataset.

    """

    def __init__(self, filename=None, feldman=None, label_name='default', favorable_classes=[1],
                 protected_attribute_names=['SEX'],
                 privileged_classes=[['male']],
                 instance_weights_name=None,
                 categorical_features=['LIMIT_BAL_CAT', 'AGE_CAT', 'EDUCATION', 'MARRIAGE'],
                 features_to_keep=[], features_to_drop=[],
                 na_values=[], custom_preprocessing=None,
                 metadata=default_mappings):
        
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            '..', 'data', 'credit.csv')

        try:
            df = pd.read_csv(filepath, na_values=na_values)
        except IOError as err:
            print("IOError: {}".format(err))
            print("To use this class, please download the following file:")
            print("\n\thttps://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv")
            print("\nand place it, as-is, in the folder:")
            print("\n\t{}\n".format(os.path.abspath(os.path.join(
               os.path.abspath(__file__), '..', '..', 'data', 'raw', 'biased_credit'))))
            import sys
            sys.exit(1)
            
        
        super(CreditDataset, self).__init__(df=df, label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop, na_values=na_values,
            custom_preprocessing=custom_preprocessing, metadata=metadata)