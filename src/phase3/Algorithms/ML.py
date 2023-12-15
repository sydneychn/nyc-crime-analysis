import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.linear_model import LinearRegression
from datetime import datetime
def sklearn_KNN(X_train,X_test,Y_train,k):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train,Y_train)
    predictions = model.predict(X_test)
    return predictions


#KNN to classify missing race values
#Read data
data = pd.read_csv('cleaneddata.csv')
unknownRace = data.loc[data['PERP_RACE'] == 'UNKNOWN'].dropna()
knownRace = data.loc[data['PERP_RACE'] != 'UNKNOWN'].dropna()
X_test = unknownRace[['AGE_GROUP','OFNS_DESC']].to_numpy()
X_train = knownRace[['AGE_GROUP','OFNS_DESC']].to_numpy()
Y_train = knownRace['PERP_RACE'].to_numpy()
k = 5
print(len(X_test))

 
# Label encoding for Age Group and Crime Description
age_group_encoder = LabelEncoder()
ofns_desc_encoder = LabelEncoder()
#Encodes labels of column 0 and 1 (age group and crime description)
X_train[:, 0] = age_group_encoder.fit_transform(X_train[:, 0])
X_test[:, 0] = age_group_encoder.transform(X_test[:, 0])
X_train[:, 1] = ofns_desc_encoder.fit_transform(X_train[:, 1])
X_test[:, 1] = ofns_desc_encoder.transform(X_test[:, 1])

# Predict criminal race using KNN
predictions = sklearn_KNN(X_train, X_test, Y_train, k)

#Declare plot size and dictionary for x and y axis
plt.figure(figsize=(11, 12))
crime_desc_labels = {}
for desc in enumerate(knownRace['OFNS_DESC'].unique()):
    crime_desc_labels[desc[0]] = desc[1]
age_group_labels = {
    0: '<18',
    1: '18-24',
    2: '25-44',
    3: '45-64',
    4: '65+' 
}
#Create scatter plot with different colors for each race
for label in set(predictions):
    mask = predictions == label
    plt.scatter(X_test[mask, 0], X_test[mask, 1], label=label, alpha=0.8)

# Get labels for x and y
plt.xticks(range(5), [age_group_labels[i] for i in range(5)])
plt.yticks(range(len(crime_desc_labels)), [crime_desc_labels[i] for i in range(len(crime_desc_labels))])
# Add labels and legend
plt.xlabel('Age Group')
plt.ylabel('Crime Description')
plt.title('Predicted Criminal Race Based on Age Group and Crime Description')
plt.legend(loc='best')

# Show the plot
plt.show()

# One-Class SVM to calculate outliers

#Selecting only features we need (lat and long)

X = data[['Latitude','Longitude']]
X.dropna(inplace=True)
X = X[(X['Latitude'] != 0) & (X['Longitude'] != 0)]
X = X.sample(1000)

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

#Logistic Regression
csvFile = pd.read_csv('NYPD_Arrest_Data__Year_to_Date_.csv')

csvFile['ARREST_DATE'] = pd.to_datetime(csvFile['ARREST_DATE']).dt.date
daily_crime_counts = csvFile.groupby('ARREST_DATE').size().reset_index(name='NUM_OF_CRIMES')

x = pd.to_datetime(daily_crime_counts['ARREST_DATE']).map(lambda x: x.toordinal()).values.reshape(-1,1)
y = daily_crime_counts['NUM_OF_CRIMES']

X_train, X_test, y_train, y_test = train_test_split(x, y)

linreg = LinearRegression()
linreg.fit(X_train, y_train)
predictions = linreg.predict(X_test)

X_test_dates = [datetime.fromordinal(int(date)) for date in X_test.flatten()]
plt.scatter(X_test_dates, y_test, color='black', label='Values')
plt.plot(X_test_dates, predictions, color='red', linewidth=3, label='Line of Best Fit')
plt.xlabel('Dates')
plt.ylabel('Number of Crimes')
plt.legend()
plt.title('Linear Regression: Number of Crimes Based on Date')
plt.show()
