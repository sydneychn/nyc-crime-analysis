import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
# One-Class SVM to calculate outliers
data = pd.read_csv('cleaneddata.csv')
sampleSize = 1000

def one_class_svm(data):
    #Selecting only features we need (lat and long)
    X = data[['Latitude','Longitude']]
    X.dropna(inplace=True)
    X = X[(X['Latitude'] != 0) & (X['Longitude'] != 0)]
    X = X.sample(10000)

    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = OneClassSVM(nu=0.05)  # You can adjust the 'nu' parameter based on your dataset
    model.fit(X_train_scaled)
    predictions = model.predict(X_test_scaled)

    plt.scatter(X_test['Latitude'], X_test['Longitude'], c='yellowgreen', label='Normal Data')

    # Identify the outliers predicted by the One-Class SVM (-1 indicates an outlier)
    outliers = X_test[predictions == -1]
    plt.scatter(outliers['Latitude'], outliers['Longitude'], c='indianred', label='Outliers')

    plt.title('One-Class SVM')
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.legend()
    plt.show()
