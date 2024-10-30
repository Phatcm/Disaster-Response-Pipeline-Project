import sys
import pandas as pd
from sqlalchemy import create_engine


def load_datasets(messages_file, categories_file):
    """
    Load and merge datasets.

    Parameters:
        messages_file (str): Path to the messages CSV file.
        categories_file (str): Path to the categories CSV file.
    
    Returns:
        DataFrame: Combined dataset.
    """
    messages = pd.read_csv(messages_file)
    categories = pd.read_csv(categories_file)
    merged_data = pd.merge(messages, categories, on='id')
    return merged_data


def refine_data(df):
    """
    Clean and structure the merged dataset.

    Parameters:
        df (DataFrame): DataFrame containing merged data.
        
    Returns:
        DataFrame: Refined dataset.
    """
    # Split the 'categories' column into individual columns
    categories_expanded = df['categories'].str.split(';', expand=True)
    
    # Extract column names for categories from the first row
    headers = categories_expanded.iloc[0].apply(lambda val: val.split('-')[0])
    
    # Rename columns in the expanded categories DataFrame
    categories_expanded.columns = headers
    
    # Convert category values to binary (0 or 1)
    categories_expanded = categories_expanded.apply(lambda col: col.str[-1].astype(int))
    
    # Replace the original 'categories' column in df with expanded categories
    df = df.drop(columns=['categories']).join(categories_expanded)
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Remove 'child_alone' column as it contains only zeroes
    if 'child_alone' in df.columns:
        df = df.drop(columns=['child_alone'])
    
    # Correct inconsistent values in 'related' column
    df['related'] = df['related'].apply(lambda x: 1 if x == 2 else x)
        
    return df


def save_to_database(df, db_filename):
    """
    Persist cleaned data to an SQLite database.

    Parameters:
        df (DataFrame): Cleaned data to save.
        db_filename (str): SQLite database filename.
        
    Returns:
        None
    """
    engine = create_engine(f'sqlite:///{db_filename}')
    df.to_sql('Disaster_Response', engine, index=False, if_exists='replace')


def main():
    try:
        if len(sys.argv) == 4:
            messages_file, categories_file, db_file = sys.argv[1:]
        else:
            messages_file = input("Enter the path to the messages file (e.g., data/disaster_messages.csv): ")
            categories_file = input("Enter the path to the categories file (e.g., data/disaster_categories.csv): ")
            db_file = input("Enter the database filename to save cleaned data (e.g., data/Disaster_Response.db): ")
        
        print(f'Loading data from:\n  Messages: {messages_file}\n  Categories: {categories_file}')
        df = load_datasets(messages_file, categories_file)

        print('Cleaning data...')
        df = refine_data(df)
        
        print(f'Saving cleaned data to database: {db_file}')
        save_to_database(df, db_file)
        print('Data successfully saved!')

    except Exception as error:
        print(f"Error accessing files or processing data: {error}")
        print('Usage example: python process_data.py disaster_messages.csv disaster_categories.csv Disaster_Response.db')


if __name__ == '__main__':
    main()
