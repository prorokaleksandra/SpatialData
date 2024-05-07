import pandas as pd

FILE_PATH = "data/IF_data/"


def standardize_phenotype(phenotype):
    markers = set()
    start = 0
    for i in range(len(phenotype)):
        if phenotype[i] == '-' or phenotype[i] == '+':
            markers.add(phenotype[start:i + 1])
            start = i + 1
    return ''.join(sorted(markers))


def get_panel(panel, patient):
    mapping = pd.read_csv(f'data/{panel}_phen_to_cell_mapping.csv', sep=',', header=0)
    mapping['phenotype'] = mapping['phenotype'].apply(lambda x: standardize_phenotype(x))

    data = pd.read_csv(FILE_PATH + f"{panel}/" + patient + f"_{panel}.csv", sep=',', header=0, index_col=0)
    data['phenotype'] = data['phenotype'].apply(lambda x: standardize_phenotype(x))
    return data.merge(mapping, on='phenotype', how='left')

