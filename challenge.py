import json
import pandas as pd
import numpy as np
# import module for regular expressions
import re
from sqlalchemy import create_engine
from config import db_password
import time

# Extract the data
# Kaggle data
file_dir = 'C:/Vandy/DataAnalyticsBootCamp/MyRepo/Movies-ETL/Resources/'
kaggle_metadata = pd.read_csv(f'{file_dir}movies_metadata.csv', low_memory = False)

# Wiki data
file_dir = 'C:/Vandy/DataAnalyticsBootCamp/MyRepo/Movies-ETL/Resources/'
with open(f'{file_dir}/wikipedia.movies.json', mode='r') as file:
    wiki_movies_raw = json.load(file)

# Ratings Data
file_dir = 'C:/Vandy/DataAnalyticsBootCamp/MyRepo/Movies-ETL/Resources/'
ratings = pd.read_csv(f'{file_dir}ratings.csv', low_memory = False)


# Function uses the above dataframes, cleans up the data, and load to SQL database
def transform_load_movie_data(wiki_movies_raw,ratings,kaggle_metadata):
    # Kaggle processing
    # Find adult movies and drop them all
    kaggle_metadata = kaggle_metadata[kaggle_metadata['adult'] == 'False'].drop('adult',axis='columns')

    # set data types for the columns
    kaggle_metadata['budget'] = kaggle_metadata['budget'].astype(int)
    kaggle_metadata['id'] = pd.to_numeric(kaggle_metadata['id'], errors='raise')
    kaggle_metadata['popularity'] = pd.to_numeric(kaggle_metadata['popularity'], errors='raise')
    kaggle_metadata['release_date'] = pd.to_datetime(kaggle_metadata['release_date'])

    # Create Boolean column of video
    kaggle_metadata['video'] = kaggle_metadata['video'] == 'True'

    # Wiki processing
    # Put raw data into dataframe
    wiki_movies_df = pd.DataFrame(wiki_movies_raw)
    wiki_movies = [movie for movie in wiki_movies_raw
               if ('Director' in movie or 'Directed by' in movie)
                   and 'imdb_link' in movie
                   and 'No. of episodes' not in movie]

    # Make another dataframe from wiki_movies list; this drops all nan columns
    wiki_movies_df = pd.DataFrame(wiki_movies)

    # add new function with the clean_movie function to change column names
    def clean_movie(movie):
        movie = dict(movie) #create a non-destructive copy
        alt_titles = {}
        # combine alternate titles into one list
        for key in ['Also known as','Arabic','Cantonese','Chinese','French',
                    'Hangul','Hebrew','Hepburn','Japanese','Literally',
                    'Mandarin','McCune-Reischauer','Original title','Polish',
                    'Revised Romanization','Romanized','Russian',
                    'Simplified','Traditional','Yiddish']:
            if key in movie:
                alt_titles[key] = movie[key]
                movie.pop(key)
            # Add new column alt_titles         
            if len(alt_titles) > 0:
                movie['alt_titles'] = alt_titles

        # merge column names
        def change_column_name(old_name, new_name):
            if old_name in movie:
                movie[new_name] = movie.pop(old_name)
        change_column_name('Adaptation by', 'Writer(s)')
        change_column_name('Country of origin', 'Country')
        change_column_name('Directed by', 'Director')
        change_column_name('Distributed by', 'Distributor')
        change_column_name('Edited by', 'Editor(s)')
        change_column_name('Length', 'Running time')
        change_column_name('Original release', 'Release date')
        change_column_name('Music by', 'Composer(s)')
        change_column_name('Produced by', 'Producer(s)')
        change_column_name('Producer', 'Producer(s)')
        change_column_name('Productioncompanies ', 'Production company(s)')
        change_column_name('Productioncompany ', 'Production company(s)')
        change_column_name('Released', 'Release Date')
        change_column_name('Release Date', 'Release date')
        change_column_name('Screen story by', 'Writer(s)')
        change_column_name('Screenplay by', 'Writer(s)')
        change_column_name('Story by', 'Writer(s)')
        change_column_name('Theme music composer', 'Composer(s)')
        change_column_name('Written by', 'Writer(s)')

        return movie

    clean_movies = [clean_movie(movie) for movie in wiki_movies]
    
    # make list a dataframe
    wiki_movies_df = pd.DataFrame(clean_movies)
    print("Wiki df type ", type(wiki_movies_df))
    print(wiki_movies_df.columns.tolist())
    # Remove Duplicate movies by checking the imdb id
    # extract the imdb id from the imdb_link. format: tt1234567
    wiki_movies_df['imdb_id'] = wiki_movies_df['imdb_link'].str.extract(r'(tt\d{7})')
    # drop duplicates
    # only consider column imdb_id and drop in place
    wiki_movies_df.drop_duplicates(subset='imdb_id', inplace=True)

    # Remove rows columns that have less than 90% null values
    wiki_columns_to_keep = [column for column in wiki_movies_df.columns if wiki_movies_df[column].isnull().sum() < len(wiki_movies_df) * 0.9]
    wiki_movies_df = wiki_movies_df[wiki_columns_to_keep]

    # Make 'Box office' column numeric with regex
    # make a data series that drops missing values
    box_office = wiki_movies_df['Box office'].dropna() 

    # Some results are a list. Combine elements in list with join. Use if statement to check if the element is a list.
    box_office = box_office.apply(lambda x: ' '.join(x) if type(x) == list else x)

    form_one = r'\$\s*\d+\.?\d*\s*[mb]illi?on'
    form_two = r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)'

    def parse_dollars(s):
        # if s is not a string, return NaN
        if type(s) != str:
            return np.nan
        # if input is of the form $###.# million
        if re.match(r'\$\s*\d+\.?\d*\s*milli?on', s, flags=re.IGNORECASE):
            # remove dollar sign, spaces, commas, and letters and " million"
            s = re.sub('\$|\s|[a-zA-Z]','', s)
            # convert to float and multiply by a million
            value = float(s) * 10**6
            # return value
            return value
        # if input is of the form $###.# billion
        elif re.match(r'\$\s*\d+\.?\d*\s*billi?on', s, flags=re.IGNORECASE):
            # remove dollar sign, spaces, commas, and letters and " billion"
            s = re.sub('\$|\s|[a-zA-Z]','', s)
            # convert to float and multiply by a billion
            value = float(s) * 10**9
            # return value
            return value
        # if input is of the form $###,###,###
        elif re.match(r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)', s, flags=re.IGNORECASE):
            # remove dollar sign and commas
            s = re.sub('\$|,','', s)
            # convert to float
            value = float(s)
            # return value
            return value
        # otherwise, return NaN
        else:
            return np.nan


    # extract the values from box_office using str.extract. 
    # Then apply parse_dollars to the first column in the DataFrame returned by str.extract
    wiki_movies_df['box_office'] = box_office.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)

    # Don't need Box office column anymore, so drop it.
    wiki_movies_df.drop('Box office', axis=1, inplace=True)

    # Section to Parse Budget data from wiki_movies_df
    # Create a budget variable
    budget = wiki_movies_df['Budget'].dropna()

    # Convert any lists to strings
    budget = budget.map(lambda x: ' '.join(x) if type(x) == list else x)

    #  remove any values between a dollar sign and a hyphen
    budget = budget.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)

    # Remove the citation references
    budget = budget.str.replace(r'\[\d+\]\s*', '')

    # Parse the budget data
    wiki_movies_df['budget'] = budget.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)

    # Don't need Budget column anymore, so drop it.
    wiki_movies_df.drop('Budget', axis=1, inplace=True)

    # Start parse the release date section
    # Make a variable to hold Release date column
    release_date = wiki_movies_df['Release date'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)

    # create forms for parsing
    date_form_one = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s[123]\d,\s\d{4}'
    date_form_two = r'\d{4}.[01]\d.[123]\d'
    date_form_three = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}'
    date_form_four = r'\d{4}'

    # Use to_datetime() method in Pandas to parse the dates
    wiki_movies_df['release_date'] = pd.to_datetime(release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})')[0], infer_datetime_format=True)

    # Section to Parse Running Time
    # Create variable to hold column
    running_time = wiki_movies_df['Running time'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)


    # only want to extract digits, and we want to allow for both possible patterns
    running_time_extract = running_time.str.extract(r'(\d+)\s*ho?u?r?s?\s*(\d*)|(\d+)\s*m')

    # convert new DataFrame's strings to numeric values
    running_time_extract = running_time_extract.apply(lambda col: pd.to_numeric(col, errors='coerce')).fillna(0)

    # convert the hour capture groups and minute capture groups to minutes if the pure minutes capture group is zero
    wiki_movies_df['running_time'] = running_time_extract.apply(lambda row: row[0]*60 + row[1] if row[2] == 0 else row[2], axis=1)


    # Drop Running time from the dataset
    wiki_movies_df.drop('Running time', axis=1, inplace=True)

    # Ratngs data Processing
    # Dates are reasonable so assign to Timestamp column
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')

    #Merge the Wiki and Kaggle files
    movies_df = pd.merge(wiki_movies_df, kaggle_metadata, on='imdb_id', suffixes=['_wiki','_kaggle'])

    # Drop any outlier where release date is way off
    movies_df = movies_df.drop(movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')].index)

    # Execute the plan
    # drop the title_wiki, release_date_wiki, Language, and Production company(s) columns
    movies_df.drop(columns=['title_wiki','release_date_wiki','Language','Production company(s)'], inplace=True)

    # make a function that fills in missing data for a column pair and then drops the redundant column
    def fill_missing_kaggle_data(df, kaggle_column, wiki_column):
        df[kaggle_column] = df.apply(
            lambda row: row[wiki_column] if row[kaggle_column] == 0 else row[kaggle_column]
            , axis=1)
        df.drop(columns=wiki_column, inplace=True)

    fill_missing_kaggle_data(movies_df, 'runtime', 'running_time')
    fill_missing_kaggle_data(movies_df, 'budget_kaggle', 'budget_wiki')
    fill_missing_kaggle_data(movies_df, 'revenue', 'box_office')

    # Reorder columns
    movies_df = movies_df.loc[:, ['imdb_id','id','title_kaggle','original_title','tagline','belongs_to_collection','url','imdb_link',
                        'runtime','budget_kaggle','revenue','release_date_kaggle','popularity','vote_average','vote_count',
                        'genres','original_language','overview','spoken_languages','Country',
                        'production_companies','production_countries','Distributor',
                        'Producer(s)','Director','Starring','Cinematography','Editor(s)','Writer(s)','Composer(s)','Based on'
                        ]]

    # Rename columns
    movies_df.rename({'id':'kaggle_id',
                    'title_kaggle':'title',
                    'url':'wikipedia_url',
                    'budget_kaggle':'budget',
                    'release_date_kaggle':'release_date',
                    'Country':'country',
                    'Distributor':'distributor',
                    'Producer(s)':'producers',
                    'Director':'director',
                    'Starring':'starring',
                    'Cinematography':'cinematography',
                    'Editor(s)':'editors',
                    'Writer(s)':'writers',
                    'Composer(s)':'composers',
                    'Based on':'based_on'
                    }, axis='columns', inplace=True)

    # Transform and Merge Rating Data
    # groupby on the “movieId” and “rating” columns and take the count for each group
    # and Rename the “userId” column to “count.”
    rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count() \
                    .rename({'userId':'count'}, axis=1) 

    # pivot this data so that movieId is the index,
    # the columns will be all the rating values, and 
    # the rows will be the counts for each rating value.
    rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count() \
                    .rename({'userId':'count'}, axis=1) \
                    .pivot(index='movieId',columns='rating', values='count')

    # Rename columns
    rating_counts.columns = ['rating_' + str(col) for col in rating_counts.columns]

    # merge the rating counts into movies_df
    movies_with_ratings_df = pd.merge(movies_df, rating_counts, left_on='kaggle_id', right_index=True, how='left')

    # fill missing ratings values with zero
    movies_with_ratings_df[rating_counts.columns] = movies_with_ratings_df[rating_counts.columns].fillna(0)

    # Load this dataframe to a database!
    # Database connection string
    db_string = f"postgres://postgres:{db_password}@127.0.0.1:5432/movie_data"

    # Create the database engine
    engine = create_engine(db_string)

    # Delete data from tables
    # Delete
    engine.execute("DELETE FROM movies")
    engine.execute("DELETE FROM movies_with_ratings")
    # engine.execute("DELETE FROM ratings")

    # Import Movie data to sql engine
    movies_df.to_sql(name='movies', con=engine, if_exists='append')


    # Import Movies_with_ratings data to sql engine
    movies_with_ratings_df.to_sql(name='movies_with_ratings', con=engine, if_exists='append')

    ## Import large ratings dataset - commented out due to long run time
    #rows_imported = 0
    ## get the start_time from time.time()
    #start_time = time.time()
    #for data in pd.read_csv(f'{file_dir}ratings.csv', chunksize=1000000):
    #    print(f'importing rows {rows_imported} to {rows_imported + len(data)}...', end='')
    #    data.to_sql(name='ratings', con=engine, if_exists='append')
    #    rows_imported += len(data)

    #    # add elapsed time to final print out
    #    print(f'Done. {time.time() - start_time} total seconds elapsed')2

# Run function to transform and load data
transform_load_movie_data(wiki_movies_raw,ratings,kaggle_metadata)

