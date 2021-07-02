# Contains the helper functions for the optim_preproc class
import numpy as np
import pandas as pd


def get_distortion_adult(vold, vnew):
    """Distortion function for the adult dataset. We set the distortion
    metric here. See section 4.3 in supplementary material of
    http://papers.nips.cc/paper/6988-optimized-pre-processing-for-discrimination-prevention
    for an example

    Note:
        Users can use this as templates to create other distortion functions.

    Args:
        vold (dict) : {attr:value} with old values
        vnew (dict) : dictionary of the form {attr:value} with new values

    Returns:
        d (value) : distortion value
    """

    # Define local functions to adjust education and age
    def adjustEdu(vnew, vold):
        distort = {}
        distort['edu'] = pd.DataFrame(
                                {'less_than_6':   [0., 1., 2.],
                                '6_to_12':        [-1., 0., 1.],
                                'greater_than_12':[-2., -1., 0.]},
                                index=['less_than_6', '6_to_12', 'greater_than_12'])
        
        return distort['edu'].loc[vnew, vold]

    def adjustAge(a):
        if a == 'less_than_25':
            return 25
        elif a == '25_to_45':
            return 35
        elif a == 'greater_than_45':
            return 45
        else:
            return float(a)

    def adjustInc(a):
        if a == "<=50K":
            return 0
        elif a == ">50K":
            return 1
        else:
            return int(a)
    
    #print(vold)
    # value that will be returned for events that should not occur
    bad_val = 3.0

    # Adjust education years
    diff = adjustEdu(vold['edu_level'], vnew['edu_level'])
    #eOld = adjustEdu(vold['edu_level'])
    #eNew = adjustEdu(vnew['edu_level'])

    # Education cannot be lowered or increased in more than 1 year
    if (diff < 0.0 or diff > 1.0):
        return bad_val

    # adjust age
    aOld = adjustAge(vold['age'])
    aNew = adjustAge(vnew['age'])

    # Age cannot be increased or decreased in more than a decade
    if np.abs(aOld-aNew) > 10.0:
        return bad_val

    # Penalty of 2 if age is decreased or increased
    if np.abs(aOld-aNew) > 0:
        return 2.0

    # Adjust income
    incOld = adjustInc(vold['income'])
    incNew = adjustInc(vnew['income'])

    # final penalty according to income
    if incOld > incNew:
        return 1.0
    else:
        return 0.0


def get_distortion_german(vold, vnew):
    """Distortion function for the german dataset. We set the distortion
    metric here. See section 4.3 in supplementary material of
    http://papers.nips.cc/paper/6988-optimized-pre-processing-for-discrimination-prevention
    for an example

    Note:
        Users can use this as templates to create other distortion functions.

    Args:
        vold (dict) : {attr:value} with old values
        vnew (dict) : dictionary of the form {attr:value} with new values

    Returns:
        d (value) : distortion value
    """

    # Distortion cost
    distort = {}
    distort['Credit_history'] = pd.DataFrame(
                                {'None/Paid': [0., 1., 2.],
                                'Delay':      [1., 0., 1.],
                                'Other':      [2., 1., 0.]},
                                index=['None/Paid', 'Delay', 'Other'])
    
    distort['Savings'] = pd.DataFrame(
                            {'Unknown/None':  [0., 1., 2.],
                            '<500':           [1., 0., 1.],
                            '500+':           [2., 1., 0.]},
                            index=['Unknown/None', '<500', '500+'])
    distort['Status'] = pd.DataFrame(
                            {'None':          [0., 1., 2.],
                            '<200':           [1., 0., 1.],
                            '200+':           [2., 1., 0.]},
                            index=['None', '<200', '200+'])
    distort['credit'] = pd.DataFrame(
                        {'Bad Credit' :    [0., 1.],
                         'Good Credit':    [2., 0.]},
                         index=['Bad Credit', 'Good Credit'])
    distort['Sex'] = pd.DataFrame(
                        {0.0:    [0., 2.],
                         1.0:    [2., 0.]},
                         index=[0.0, 1.0])
    distort['Age'] = pd.DataFrame(
                                {'less_than_25':    [0., 1., 2.],
                                '25_to_45':         [1., 0., 1.],
                                'greater_than_45':  [2., 1., 0.]},
                                index=['less_than_25', '25_to_45', 'greater_than_45'])
    
    #print(vold)
    total_cost = 0.0
    for k in vold:
        if k in vnew:
            total_cost += distort[k].loc[vnew[k], vold[k]]

    return total_cost

def get_distortion_compas(vold, vnew):
    """Distortion function for the compas dataset. We set the distortion
    metric here. See section 4.3 in supplementary material of
    http://papers.nips.cc/paper/6988-optimized-pre-processing-for-discrimination-prevention
    for an example

    Note:
        Users can use this as templates to create other distortion functions.

    Args:
        vold (dict) : {attr:value} with old values
        vnew (dict) : dictionary of the form {attr:value} with new values

    Returns:
        d (value) : distortion value
    """
    # Distortion cost
    distort = {}
    distort['two_year_recid'] = pd.DataFrame(
                                {'Did recid.':     [0., 2.],
                                 'No recid.':     [2., 0.]},
                                index=['Did recid.', 'No recid.'])
    distort['Age'] = pd.DataFrame(
                            {'less_than_25':    [0., 1., 2.],
                            '25_to_45':         [1., 0., 1.],
                            'greater_than_45':  [2., 1., 0.]},
                            index=['less_than_25', '25_to_45', 'greater_than_45'])
 
    distort['Prior'] = pd.DataFrame(
                            {'less_than_3':     [0., 1., 2.],
                            '3_to_10':          [1., 0., 1.],
                            'greater_than_10':  [2., 1., 0.]},
                            index=['less_than_3', '3_to_10', 'greater_than_10'])
    distort['Sex'] = pd.DataFrame(
                        {0.0:    [0., 2.],
                         1.0:    [2., 0.]},
                         index=[0.0, 1.0])
    distort['Race'] = pd.DataFrame(
                        {0.0:    [0., 2.],
                         1.0:    [2., 0.]},
                         index=[0.0, 1.0])

    total_cost = 0.0
    #print(vold)
    for k in vold:
        if (k in ['length_of_stay']):
            continue
        if k in vnew:
            total_cost += distort[k].loc[vnew[k], vold[k]]

    return total_cost

