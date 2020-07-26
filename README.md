# Movies-ETL
Perform automated ETL (Extract-Transform-Load) on movie data files from multiple sources.  Data is loaded to PostgreSQL database within a Python script. The database will be used for a 

## Background and Results
Vanderbilt Data Analysis Bootcamp
Module 8 Challenge

A large online retailer, Amazing Prime, is holding a hackathon to help them predict which low-budget movie releases will be popular so they can buy the streaming rights at a great price.  Our goal is to import movie data, clean up and transform the data, and load to a database.  We perform multiple cleanup steps including, finding and removing bad data, setting correct data types, merging files, dropping unnecessary columns, and adding aggregated rating information. All this ETL occurs within a Python script.


### Resources
Using three csv files provided, build a table that cleaned up movie information from which we can provide reports for management.

Data Sources:

- wikipedia.movies.json - Wikipedia Movie data
- movies_metadata.csv - Kaggle Movie Metadata
- ratings.csv - MovieLens rating data (from Kaggle)

Software: 

- Pyton 3.7.6
- Jupyter Notebook 6.0.3
- Pandas 1.0.1
- Numpy 1.18.1
- sqlalchemy 1.3.18
- Postgres database
- PgAdmin editor
- VS Code

### Results
Three tables were created
movies - the merged files from wiki and kaggle: 6051 rows
movies_with_ratings - the merged files from wiki, kaggle and an aggregated ratings count: 6051 rows
ratings - the entire ratings file with over 26 million rows

## Assumptions
Spots where code could be added to handle possible errors:  
- if the files don't exist
- if the database doesn't exist

The final database should be reviewed for:
- if the data input files have changed
- if data is no longer available
- if there are odd outliers
- there are some new data formats that could easily be converted instead of dropping the row