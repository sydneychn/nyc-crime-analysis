# nyc-crime-analysis
# Data Analysis Project - NYC Crime Analysis
# Problem Statement
The issue of crime in New York City is a matter of significant concern due to its impact on the well-being of the residents across the city's boroughs. With a population of approximately 8.4 million residents, it is essential to understand the dynamics of crime in NYC to improve public safety, economic development, and address inequality issues.

This project aims to analyze the dataset of crimes committed in New York City from January to June 2023, focusing on the following aspects:

Identifying which borough has the most crimes committed.
Investigating correlations between the age group, sex, and race of the perpetrator and the type of crime.
Determining if there are specific times or dates when crimes are more likely to occur.
Identifying common types of crimes in different boroughs.
Analyzing geographic clusters of crimes based on latitude and longitude.
The insights gained from this analysis will help reform policy decisions, enhance public safety, and contribute to a safer environment for the residents of New York City.

# Data Sources
NYPD Arrest Data Year to Date - https://catalog.data.gov/dataset/nypd-arrest-data-year-to-date
NYPD Arrests Data Historic - https://catalog.data.gov/dataset/nypd-arrests-data-historic

# Data Cleaning/Processing
To prepare the dataset for analysis, the following data cleaning and processing steps were performed:

Removed rows with null values in the crime description and crime type columns.
Removed unnecessary columns, such as X and Y coordinates (latitude and longitude were retained).
Ensured there were no duplicates in the arrest key column.
Converted the date column into a standard DateTime data type.
Removed double quotes from the police department description (PD_DESC) column.
Converted latitude and longitude into floats for ease of analysis.
Grouped all "HISPANICS" together under "PERP_RACE" for generality.
The cleaned data was saved as a new CSV file for further analysis.

# Exploratory Data Analysis (EDA)
An exploratory data analysis was conducted to understand the dataset better and to determine the features relevant for analysis. Key findings from the EDA phase include:
The dataset contains a total of 112,105 rows.
Nine features were selected for EDA, including ARREST_DATE, 
PD_DESC, 
OFNS_DESC, 
ARREST_BORO, 
AGE_GROUP, 
PERP_SEX, 
PERP_RACE, 
Latitude, and 
Longitude.
Certain features, including 
ARREST_KEY, 
PD_CD, 
KY_CD, 
LAW_CODE, 
LAW_CAT_CD, 
ARREST_PRECINCT, and 
JURISDICTION_CODE, were excluded from the analysis for the time being.
The dataset covers the period from January 2023 to June 2023, spanning five months.

# Project Goals
The primary goals of this data analysis project are as follows:

To provide insights and trends that can inform policy decisions, enhance public safety, and contribute to a safer environment for the residents of New York City.
To identify areas with high crime rates and develop targeted crime prevention strategies.
To optimize emergency response planning and resource allocation based on crime patterns and locations.
To determine where law enforcement resources should be allocated for maximum impact.
By achieving these goals, we aim to contribute to the safety and well-being of the residents of New York City and support its economic development.
