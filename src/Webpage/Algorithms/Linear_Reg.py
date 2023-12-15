import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from datetime import datetime
#Logistic Regression
#data = pd.read_csv('NYPD_Arrest_Data__Year_to_Date_.csv')
def linear_reg(data): 
    data['ARREST_DATE'] = pd.to_datetime(data['ARREST_DATE']).dt.date
    daily_crime_counts = data.groupby('ARREST_DATE').size().reset_index(name='NUM_OF_CRIMES')

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