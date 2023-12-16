# Data Analysis Project - NYC Crime Analysis
# Problem Statement
The issue of crime in New York City is a matter of significant concern due to its impact on the well-being of the residents across the city's boroughs. With a population of approximately 8.4 million residents, it is essential to understand the dynamics of crime in NYC to improve public safety, economic development, and address inequality issues.

This project aims to analyze the dataset of crimes committed in New York City from January to June 2023, focusing on the following aspects:
- Identifying which borough has the most crimes committed.
- Investigating correlations between the age group, sex, and race of the perpetrator and the type of crime.
- Determining if there are specific times or dates when crimes are more likely to occur.
- Identifying common types of crimes in different boroughs.
- Analyzing geographic clusters of crimes based on latitude and longitude.

The insights gained from this analysis will help reform policy decisions, enhance public safety, and contribute to a safer environment for the residents of New York City.

## Data Sources
- NYPD Arrest Data Year to Date - https://catalog.data.gov/dataset/nypd-arrest-data-year-to-date
- NYPD Arrests Data Historic - https://catalog.data.gov/dataset/nypd-arrests-data-historic

# Data Cleaning/Processing
To prepare the dataset for analysis, the following data cleaning and processing steps were performed:

- Removed rows with null values in the crime description and crime type columns.
- Removed unnecessary columns, such as X and Y coordinates (latitude and longitude were retained).
- Ensured there were no duplicates in the arrest key column.
- Converted the date column into a standard DateTime data type.
- Removed double quotes from the police department description (PD_DESC) column.
- Converted latitude and longitude into floats for ease of analysis.
- Grouped all "HISPANICS" together under "PERP_RACE" for generality.
The cleaned data was saved as a new CSV file for further analysis.

## Exploratory Data Analysis (EDA)
An exploratory data analysis was conducted to understand the dataset better and to determine the features relevant for analysis. Key findings from the EDA phase include:
The dataset contains a total of 112,105 rows.
Nine features were selected for EDA, including:
- ARREST_DATE
- PD_DESC 
- OFNS_DESC 
- ARREST_BORO
- AGE_GROUP 
- PERP_SEX
- PERP_RACE
- Latitude 
- Longitude
  
Certain features, including 
ARREST_KEY, 
PD_CD, 
KY_CD, 
LAW_CODE, 
LAW_CAT_CD, 
ARREST_PRECINCT, and 
JURISDICTION_CODE, were excluded from the analysis for the time being.
The dataset covers the period from January 2023 to June 2023, spanning five months.

# Machine Learning Algorithms and Statistical Models Used
- **K-Nearest Neighbors (KNN)**: Addressed missing values in 'PERP_RACE' using KNN classification using crime type and age group, improving dataset comprehensiveness and providing insights into crime prevalence across different racial groups.
- **One-Class SVM**: Identified outlier crime locations using One-Class SVM, aiding law enforcement in efficiently allocating resources and focusing on areas requiring more attention.
- **KMeans**: Categorized crimes into borough-specific clusters with KMeans, offering valuable geographic insights for law enforcement and policymakers to understand crime patterns in different areas.
- **Support Vector Machine (SVM)**: Classified distinct crime types based on offense descriptions and days of the week, providing insights for law enforcement to address specific crime types on particular days.
- **Linear Regression**: Applied linear regression to understand the relationship between the number of crimes over time, serving as a valuable tool in understanding the dynamics of crime in NYC and developing effective methods to prevent it.
- **Chi-Square**: Used Chi-square testing to assess the association between crime categories and gender, providing insights into how gender may impact the occurrence of crimes and informing potential policy reforms.

# Webpage for Crime Dataset
We developed a user-friendly webpage enabling users to upload a cleaned NYC crime dataset. Users can select from a variety of algorithms to analyze the dataset, generating insightful results. This allows users to gain a deep understanding of crime behavior trends and patterns within the dataset provided.
### To run the webpage on your local machine, follow these quick instructions:
1. Install Flask on your machine by running the following in the terminal: pip install flask.
2. Navigate to the src directory (.../src/phase3/).
3. Run the app.py script by running the following in the terminal: python app.py.
4. Click the link generated from the output (terminal) as the webpage will be hosted locally.
   
### How to Use the Webpage:
1. To begin, select your file by clicking on "Choose File," which will navigate you to your file system. Pick a CSV file from your local system.
2. Next, opt for your preferred machine learning algorithm from a provided list of six options. Depending on your goals some algorithms will benefit you more than others. For example, if you want where the most crime is taking place you might choose Kmeans instead of SVM.
3. After making your selection, click "Show me!" and patiently await the generation of plots. 
Note: Processing times may vary, with certain algorithms, such as SVM and Kmeans, taking longer than others. Additionally, there is an option to make the webpage a dark mode so it is less stressful on your eyes.

## Languages and Tools Used:
### Programming Languages
- _Python_: Used for data processing, analysis, and the implementation of machine learning algorithms.
### Libraries and Frameworks
- _Pandas_: Employed for efficient data manipulation and analysis.
- _Matplotlib and NumPy_: Used for data visualization and numerical operations.
- _Scikit-Learn (sklearn)_: Utilized for implementing machine learning algorithms.
- _Flask_: Facilitated the development of our webpage, which performs various machine learning algorithms on CSV data files.
### Web Development
_HTML, CSS_: Utilized for creating an intuitive and responsive user interface.

