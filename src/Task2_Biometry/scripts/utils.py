import os
import numpy as np
import pandas as pd

# Load the biometry measurements from the TSV file
def load_biometry_data(dataset_path):
    biometry_file_path = os.path.join(dataset_path, 'biometry.tsv')
    biometry_df = pd.read_csv(biometry_file_path, sep='\t')

    # Rename the 'Unnamed: 0' column to 'Subject'
    biometry_df.rename(columns={'Unnamed: 0': 'Subject'}, inplace=True)

    # Drop any unnamed columns that might have been included
    biometry_df = biometry_df.loc[:, ~biometry_df.columns.str.contains('^Unnamed')]

    return biometry_df
