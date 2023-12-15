import pandas as pd

def clean(data):
    data.dropna(subset=['OFNS_DESC', 'PD_DESC', 'PD_CD','KY_CD'], inplace= True)
    data['ARREST_DATE'] = pd.to_datetime(data['ARREST_DATE'], format='%m/%d/%Y')
    data.drop('X_COORD_CD', axis=1, inplace=True)
    data.drop('Y_COORD_CD', axis=1, inplace=True)
    data.drop('New Georeferenced Column', axis=1, inplace=True)
    data.drop_duplicates(subset=['ARREST_KEY'], inplace=True)
    data['PD_DESC'] = data['PD_DESC'].str.replace('"', '')
    data['Latitude'] = data['Latitude'].astype(float)
    data['Longitude'] = data['Longitude'].astype(float)
    replacedict = {'WHITE HISPANIC': 'HISPANIC', 'BLACK HISPANIC': 'HISPANIC'}
    data['PERP_RACE'].replace(replacedict, inplace=True)
    return data