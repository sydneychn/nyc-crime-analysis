import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import svm
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

def SklearnSVM(X_train,Y_train,X_test):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray

    :rtype: numpy.ndarray 
    """

    classifier = svm.SVC(kernel='linear')
    classifier.fit(X_train,Y_train)
    Y_pred = classifier.predict(X_test)
    return Y_pred

# data = pd.read_csv('NYPD_Arrest_Data__Year_to_Date_.csv')
# data.dropna(subset=['OFNS_DESC', 'PD_DESC', 'PD_CD','KY_CD'], inplace= True)

# data['ARREST_DATE'] = pd.to_datetime(data['ARREST_DATE'], format='%m/%d/%Y')

# data.drop('X_COORD_CD', axis=1, inplace=True)
# data.drop('Y_COORD_CD', axis=1, inplace=True)
# data.drop('New Georeferenced Column', axis=1, inplace=True)


# data.drop_duplicates(subset=['ARREST_KEY'], inplace=True)


# data['PD_DESC'] = data['PD_DESC'].str.replace('"', '')

# data['Latitude'] = data['Latitude'].astype(float)
# data['Longitude'] = data['Longitude'].astype(float)

# replacedict = {'WHITE HISPANIC': 'HISPANIC', 'BLACK HISPANIC': 'HISPANIC'}

# data['PERP_RACE'].replace(replacedict, inplace=True)

# print(data)

# data.to_csv('cleaneddata.csv')


def SklearnPCA(data, components):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    # Standardize the data
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data)

    # Apply PCA
    pca = PCA(n_components=components)
    reduced_data = pca.fit_transform(standardized_data)
    return reduced_data


def testSVMWithCSV(data):
    # Load the dataset
    data = data.sample(frac=0.5, random_state=42)
    # Extract day of the week from 'ARREST_DATE' column
    data['ARREST_DATE'] = pd.to_datetime(data['ARREST_DATE'])
    data['Day_of_Week'] = data['ARREST_DATE'].dt.dayofweek

    # Encode the 'OFNS_DESC' column into numerical labels
    label_encoder = LabelEncoder()
    data['Crime_Label'] = label_encoder.fit_transform(data['OFNS_DESC'])

    # Features (X) and target variable (y)
    X = data[['Day_of_Week']]
    y = data['Crime_Label']

    # Drop rows with any NaN values in X and y
    non_nan_indices = ~np.isnan(X).any(axis=1)
    X = X[non_nan_indices]
    y = y[non_nan_indices]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred = SklearnSVM(X_train, y_train, X_test)

    # Decode numerical labels back to crime types for visualization
    predicted_crime_types = label_encoder.inverse_transform(y_pred)

    # Create a DataFrame with actual and predicted crime labels
    results = pd.DataFrame({'Actual_Crime_Type': data.loc[y_test.index, 'OFNS_DESC'], 
                            'Predicted_Crime_Type': predicted_crime_types,
                            'Day_of_Week': X_test['Day_of_Week']})

    # Mapping dictionary for day of the week labels
    day_of_week_mapping = {
        0: 'Monday',
        1: 'Tuesday',
        2: 'Wednesday',
        3: 'Thursday',
        4: 'Friday',
        5: 'Saturday',
        6: 'Sunday'
    }

    # Map day of the week labels to strings in the results DataFrame
    results['Day_of_Week'] = results['Day_of_Week'].map(day_of_week_mapping)

    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    results['Day_of_Week'] = pd.Categorical(results['Day_of_Week'], categories=days_order, ordered=True)

    # Visualize actual crime labels for specific days of the week
    plt.figure(figsize=(12, 8))
    sns.countplot(x='Day_of_Week', hue='Actual_Crime_Type', data=results)
    plt.xlabel('Day of the Week')
    plt.ylabel('Count of Crimes')
    plt.title('Actual Crime Types by Day of the Week')
    plt.legend(title='Crime Type', loc='upper right')

    predicted_crime_counts = results.groupby(['Predicted_Crime_Type', 'Day_of_Week']).size().unstack()
    predicted_crime_counts_long = predicted_crime_counts.stack().reset_index(name='Count')

    # Plot the bar chart with the correct legend
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Day_of_Week', y='Count', hue='Predicted_Crime_Type', data=predicted_crime_counts_long, palette="viridis")
    plt.title('Predicted Crime Types by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Predicted Crime Type', bbox_to_anchor=(1, 1))

    plt.show()