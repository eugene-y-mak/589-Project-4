import numpy as np


# gets all possible unique values from given attribute header/name in full dataset. Mainly for categorical
def get_attribute_values(master_dataset, label_header):
    return np.unique(master_dataset.loc[:, label_header])  # guaranteed to be sorted!
