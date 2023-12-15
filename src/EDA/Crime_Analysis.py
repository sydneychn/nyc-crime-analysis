import pandas as pd
import math



#Reading csv file
data = pd.read_csv('src/cleaneddata.csv')
print(data.head())

borough_data = data['ARREST_BORO'].value_counts().nlargest(5)
print(borough_data)
borough_data = borough_data.to_frame()
crime = borough_data.iloc[:, 0].tolist() 
pop = {
    'Brooklyn': 2736074,
    'Bronx': 1472654,
    'Manhattan': 1694251,
    'Queens': 2405464,
    'Staten Island': 495747
}
# # x = 0
# # for k,v in pop.items():
# #     print(str(k) + " - Population : " + str(v) + ", Crimes Commited: " + str(crime[x]))
# #     x += 1
# # x = 0

# # for k,v in pop.items():
# #     print(str(k) + " - Crimes per 1000 People: " + str(crime[x]/v*1000)) 
# #     x += 1

boroughs = list(pop.keys())
populations = list(pop.values())
crimes_per_1000 = [c / p * 1000 for c, p in zip(crime, populations)]

borough_data = {
    'Borough': boroughs,
    'Population': populations,
    'Total Crimes': crime,
    'Crimes per 1000 People': crimes_per_1000
}

borough_data = pd.DataFrame(borough_data)

print("Top 20 Most Common Crimes in NYC")
print(data['OFNS_DESC'].value_counts().nlargest(20).to_string(header=False) + "\n")
print("Number of Crimes Committed by Borough")
print(borough_data.to_string(index=False))
print(borough_data.describe().iloc[: , 1:])
print("Total Crimes Commited by Sex")
print(data['PERP_SEX'].value_counts().nlargest(10).to_string(header=False) + "\n")

sexbyCrime = data.groupby(['PERP_SEX','OFNS_DESC']).size().reset_index(name = 'Freq')
sexbyCrime = sexbyCrime.sort_values(by=['PERP_SEX', 'Freq'], ascending=[True,False])
topMale = sexbyCrime[sexbyCrime['PERP_SEX'] == 'M'].head(10)
topFemale = sexbyCrime[sexbyCrime['PERP_SEX'] == 'F'].head(10)