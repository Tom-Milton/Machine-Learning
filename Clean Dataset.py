import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns


def extract_label_data(df):
    """Extracts rows that satisfy the project investigation"""

    # Splits dataset into positive and negative labels for time until hospitalisation
    # No contradictory positive cases exist as there is no way to tell if someone did NOT go to hospital
    positive_labels = df['date_onset_symptoms'].notnull() & df['date_admission_hospital'].notnull()
    negative_labels = df['date_onset_symptoms'].notnull() & df['date_admission_hospital'].isnull()
    print(sum(positive_labels), 'positive labels')
    print(sum(negative_labels), 'negative labels')

    # Corrects negative_labels by removing contradictory cases (cases that say they both did and didn't go to hospital)
    # Under the assumption that no hospital outcome or death_or_discharge date implies they weren't hospitalised
    true_negative_labels = negative_labels & df['outcome'].isnull() & df['date_death_or_discharge'].isnull()
    print(sum(negative_labels) - sum(true_negative_labels), 'contradictory negative cases removed', '\n')

    # Must satisfy one of the two conditions
    df = df[positive_labels | true_negative_labels]

    return df


def calculate_date_difference(d1, d2, allow_negatives=True, allow_nan=True):
    """Calculates the number of days between two dates in the context of our dataframe"""

    def days_between(d1, d2):
        """Function to calculate the number of days between two dates"""
        d1 = datetime.strptime(d1, '%d.%m.%Y')
        d2 = datetime.strptime(d2, '%d.%m.%Y')
        return (d2 - d1).days

    # Corrects dates with the form 'date - date'
    # Chooses the first date as the correct date
    if '-' in str(d1):
        d1 = d1[:10]
    if '-' in str(d2):
        d2 = d2[:10]

    try:
        # Calculates days between onset of symptoms and admission to hospital
        days = days_between(d1, d2)

        # Flags negative values to be removed if they are set to be removed by parameter
        # We are only predicting positive time from symptom development
        if days < 0 and allow_negatives == False:
            days = 'remove'

    # If there is an error in the date formatting, flag it to be removed
    except ValueError:
        days = 'remove'

    # If a type error occurs, one of the dates is nan
    except TypeError:
        days = np.inf

        # Sets missing values to be dealt with later if set by parameter
        if not allow_nan:
            days = np.nan

    return days


def relabel_dates(df):
    """Creates label 'days_before_hospitalisation' and 'days_before_confirmation'"""

    days_before_hospitalisation = []
    days_before_confirmation = []

    # Calculates the number of days between pairs of features shown
    for a, b, c in zip(df['date_onset_symptoms'], df['date_admission_hospital'], df['date_confirmation']):
        days_before_hospitalisation.append(calculate_date_difference(a, b, allow_negatives=False))
        days_before_confirmation.append(calculate_date_difference(a, c, allow_nan=False))

    # Add series to dataframe
    df['days_before_hospitalisation'] = days_before_hospitalisation
    df['days_before_confirmation'] = days_before_confirmation

    # Remove the three original features
    df.drop(['date_onset_symptoms', 'date_admission_hospital', 'date_confirmation'], axis=1, inplace=True)

    # Remove cases flagged to be removed
    df = df[df['days_before_hospitalisation'] != 'remove']
    df = df[df['days_before_confirmation'] != 'remove']

    print(df[df['days_before_hospitalisation'] != np.inf].shape[0], 'positive labels after relabeling dates')
    print(df[df['days_before_hospitalisation'] == np.inf].shape[0], 'negative labels after relabeling dates', '\n')

    return df


def relabel_age(df):
    """Relabels age feature inaccuracies to make the values consistent.
        We use a random value from a uniform distribution to replace age ranges"""

    # Initialises new age feature
    new_age = list(df['age'])

    for i in range(len(new_age)):
        age = str(new_age[i])

        # e.g. '40-49'
        if '-' in age:
            start, end = age.split('-')
            new_age[i] = np.random.randint(int(start), int(end)+1)

        # Represents a baby between 0 and 2
        # e.g. '18 months'
        if 'month' in age:
            new_age[i] = np.random.randint(0, 3)

        # Represents a baby between 0 and 2
        # e.g. '0.25'
        if '.' in age:
            new_age[i] = np.random.randint(0, 3)

        # only '80+', 103 is the maximum age in our dataset
        if '+' in age:
            new_age[i] = np.random.randint(80, 104)

    # Assigns new age feature to df
    df['age'] = new_age

    return df


def missing_data(df):
    """Removes missing data in dataframe"""

    # Calculates number of missing cells in each column
    print('percent of entries missing in column:')
    print(round(df.isna().sum() * 100 / len(df), 2), '\n')
    original_columns = df.columns

    # Prevents exact correlation with label
    df.drop(columns='ID', inplace=True)
    # Large amount of missing data hidden as 'False'
    df.drop(columns=['chronic_disease_binary', 'travel_history_binary'], inplace=True)
    # Reflects the 'admin' part of the df which contains technical details of the location of infection
    df.drop(columns=['geo_resolution', 'country_new', 'admin_id'], inplace=True)
    print('manually removed features:', [i for i in original_columns if i not in df.columns])
    original_columns = df.columns

    # Drop missing columns
    # Must have 80% of non NA values, this threshold was confirmed by checking manually
    df.dropna(axis=1, thresh=df.shape[0] * 0.8, inplace=True)
    print('removed features for missing data:', [i for i in original_columns if i not in df.columns])
    print('remaining columns:', list(df.columns), '\n')

    # Fill some missing values to retain as many positive labels as possible
    print('number of missing rows in remaining features:')
    print(df.isna().sum(), '\n')
    df['city'].fillna(0, inplace=True)

    # Drop missing rows
    df.dropna(axis=0, inplace=True)
    print(df.shape[0], 'rows remaining')

    print(df[df['days_before_hospitalisation'] != np.inf].shape[0], 'positive labels remaining after removing missing rows')
    print(df[df['days_before_hospitalisation'] == np.inf].shape[0], 'negative labels remaining after removing missing rows', '\n')

    return df


def one_hot_encode(df):
    """One hot encode and remove sparse features"""

    # Convert some features to numeric so they aren't encoded later
    df['days_before_hospitalisation'] = pd.to_numeric(df['days_before_hospitalisation'])
    df['days_before_confirmation'] = pd.to_numeric(df['days_before_confirmation'])
    df['age'] = pd.to_numeric(df['age'])
    df['latitude'] = pd.to_numeric(df['latitude'])
    df['longitude'] = pd.to_numeric(df['longitude'])

    # Initialise encoded dataframe
    encoded_df = pd.DataFrame()
    for i in df.columns:

        # If feature is object, create dummy variables and add to new dataframe
        if df[i].dtype == 'object':
            dummies = pd.get_dummies(df[i], prefix=i)
            encoded_df = pd.concat([encoded_df, dummies], axis=1)

        # If feature is already numeric, add to new dataframe
        else:
            encoded_df[i] = df[i]

    # Drop columns with more than 99% zeros
    encoded_df = encoded_df.loc[:, (encoded_df == 0).mean() < 0.99]

    return encoded_df


def label_encode(df):
    """Label encode and create correlation matrix"""

    # Convert labels to numeric so they aren't encoded later
    df['days_before_hospitalisation'] = pd.to_numeric(df['days_before_hospitalisation'])
    df['days_before_confirmation'] = pd.to_numeric(df['days_before_confirmation'])
    df['age'] = pd.to_numeric(df['age'])
    df['latitude'] = pd.to_numeric(df['latitude'])
    df['longitude'] = pd.to_numeric(df['longitude'])

    # Label encodes object columns
    le = LabelEncoder()
    for i in df.columns:
        if df[i].dtype == 'object':
            df[i] = le.fit_transform(df[i].astype(str))

    # Produces correlation Matrix
    fig, ax = plt.subplots(figsize=(15, 10))
    corr = df.corr()
    sns.heatmap(corr, cmap='Blues', annot_kws={'size': 30}, ax=ax)
    ax.set_title("Correlation Matrix", fontsize=14)
    plt.show()

    return df


def normalise_df(df):
    """Normalises df to between 0 and 1"""

    # Extracts the label and resets the index so it can be later merged with the df
    label = df['days_before_hospitalisation']
    label.reset_index(inplace=True, drop=True)

    # Remove label and retain columns
    df.drop('days_before_hospitalisation', axis=1, inplace=True)
    cols = df.columns

    # Performs min-max scaling
    scaler = MinMaxScaler()
    normalised_df = pd.DataFrame(scaler.fit_transform(df))  # Also resets the indices

    # Rename columns and re-add label
    normalised_df.columns = cols
    normalised_df['days_before_hospitalisation'] = label

    return normalised_df


def split_df(df):
    """Splits the dataframe into a classification and a regression df.
        Cleans the respective dfs by dropping dummy features with many missing values"""

    # Calculates the regression sub-df
    reg_df = df[df['days_before_hospitalisation'] != np.inf]

    # Creates new label for the classification df
    label = np.array(df['days_before_hospitalisation'].values.tolist())  # Extract label
    clas_df = df.drop('days_before_hospitalisation', axis=1)  # Drop label
    clas_df['days_before_hospitalisation_binary'] = np.where(label == np.inf, 0, 1).tolist()  # Add binary label

    return clas_df, reg_df


def main():
    df = pd.read_csv('latestdata.csv', dtype='str')
    print('original dataframe has shape', df.shape, '\n')

    df = extract_label_data(df)
    print('dataframe satisfying the label data has shape', df.shape, '\n')

    df = relabel_dates(df)
    df = relabel_age(df)
    print('dataframe after relabeling features has shape', df.shape, '\n')

    df = missing_data(df)
    print('dataframe with missing columns and rows removed has shape', df.shape, '\n')

    # df = label_encode(df)
    df = one_hot_encode(df)
    print('encoded dataframe has shape', df.shape, '\n')

    df = normalise_df(df)
    print('normalised dataframe has shape', df.shape, '\n')

    clas_df, reg_df = split_df(df)
    print('classification dataframe has shape', clas_df.shape)
    print('regression dataframe has shape', reg_df.shape, '\n')

    clas_df.to_csv('cleaned classification dataset.csv', index=False)
    reg_df.to_csv('cleaned regression dataset.csv', index=False)


main()
