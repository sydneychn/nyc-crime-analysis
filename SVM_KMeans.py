import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.metrics import confusion_matrix



def nearestCentroid(centroids, point):
    distances = []
    for centroid in centroids:
        dist = np.linalg.norm(centroid - point)
        distances.append(dist)
    min_index = np.argmin(distances)
    return min_index #returns index of minimum centroid in original list of centroids


def createClusters(centroids, points):
    clusters =  [[] for _ in centroids]
    for point in points:
        nearestCentroidIndex = nearestCentroid(centroids,point)
        clusters[nearestCentroidIndex].append(point)
    clusters = [np.array(cluster) for cluster in clusters] #convert all lists to ndarrays
    return clusters


def newCentroids (centroids, clusters):
    ret_val = []
    for cluster in clusters:
        mean = np.mean(cluster, axis=0)
        ret_val.append(mean)
    return np.array(ret_val)


def Kmeans(X_train, K):
    #TO-DO PART 1
    """
    :type X_train: numpy.ndarray
    :type N: int
    :type K: int
    :rtype: List[numpy.ndarray]
    """
    numOfClusters = K

    centroids = X_train.copy()
    np.random.shuffle(centroids)
    centroids = centroids[:numOfClusters]

    while True:
        clusters = createClusters(centroids, X_train)

        new_centroids = newCentroids(centroids,clusters)

        #if converge return the clusters
        if np.array_equal(new_centroids, centroids):
            return clusters
        
        centroids = new_centroids.copy()

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


def testKMeansWithCSV():
    df = pd.read_csv('cleaneddata.csv')
    filtered_df = df[(df['Latitude'] != 0) & (df['Longitude'] != 0)]
    last_two_columns_array = filtered_df.iloc[:, -2:].values
    print(last_two_columns_array)


    K = 5
    result_clusters = Kmeans(last_two_columns_array, K)

    # Calculate the minimum and maximum values for x and y coordinates in your data
    min_x, max_x = np.min(last_two_columns_array[:, 0]), np.max(last_two_columns_array[:, 0])
    min_y, max_y = np.min(last_two_columns_array[:, 1]), np.max(last_two_columns_array[:, 1])

    # Set the x and y axis limits to zoom in on the data
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)

    # Plot the clusters and centroids
    colors = ['red', 'green', 'blue', 'purple', 'orange']
    for i, cluster in enumerate(result_clusters):
        plt.scatter(cluster[:, 0], cluster[:, 1], color=colors[i], label=f'Cluster {i+1}')

    # Plot centroids
    centroids = np.array([np.mean(cluster, axis=0) for cluster in result_clusters])
    plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='X', s=200, label='Centroids')

    plt.title('Clusters of Crime based on Geolocation')
    plt.xlabel('latitude')
    plt.ylabel('longitude')
    plt.legend()
    plt.show()



def testSVMWithCSV():
    # Load the dataset
    data = pd.read_csv('cleaneddata.csv')
    # data = data.sample(frac=0.01, random_state=42)
    # Extract day of the week from 'ARREST_DATE' column
    data['ARREST_DATE'] = pd.to_datetime(data['ARREST_DATE'])
    data['Day_of_Week'] = data['ARREST_DATE'].dt.dayofweek

    # Encode the 'OFNS_DESC' column into numerical labels
    label_encoder = LabelEncoder()
    data['Crime_Label'] = label_encoder.fit_transform(data['OFNS_DESC'])

    # Features (X) and target variable (y)
    X = data[['Day_of_Week']]
    y = data['Crime_Label']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    y_pred = SklearnSVM(X_train, y_train, X_test)

    # Decode numerical labels back to crime types for visualization
    predicted_crime_types = label_encoder.inverse_transform(y_pred)

    # Create a DataFrame with actual and predicted crime labels
    results = pd.DataFrame({'Actual_Crime_Type': data.loc[y_test.index, 'OFNS_DESC'], 
                            'Predicted_Crime_Type': label_encoder.inverse_transform(y_pred),
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

    # Visualize actual vs. predicted crime labels for specific days of the week
    plt.figure(figsize=(12, 8))
    sns.countplot(x='Day_of_Week', hue='Actual_Crime_Type', data=results)
    plt.xlabel('Day of the Week')
    plt.ylabel('Count of Crimes')
    plt.title('Actual Crime Types by Day of the Week')
    plt.legend(title='Crime Type', loc='upper right')
    plt.show()

testKMeansWithCSV()
# testSVMWithCSV()