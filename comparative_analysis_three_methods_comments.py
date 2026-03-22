"""
Comparative Analysis with Enhanced Documentation

Extended version of comparative_analysis_three_methods.py with additional
comments and group alias transformations for publication.
"""

import pandas as pd

df = pd.read_csv('/mnt/c/Users/paula.sanchez/Documents/TomolabDensityMeasurements/Results/combined_results.csv')

mapping = {'PV': 'OBC', 'MG': 'MGA', 'LUB': 'MGB'}
df['sample_group'] = df['sample_group'].replace(mapping)

# Create sample_alias column properly grouping by both sample_group and sample
df['sample_order'] = df.groupby('sample_group')['sample'].transform(lambda x: pd.factorize(x)[0])
df['sample_alias'] = df['sample_group'] + '_' + df['sample_order'].astype(str)

df.drop(columns=['sample_order'], inplace=True)

output_filename = '/mnt/c/Users/paula.sanchez/Documents/TomolabDensityMeasurements/Results/combined_results_after_comments.csv'
df.to_csv(output_filename, index=False)

