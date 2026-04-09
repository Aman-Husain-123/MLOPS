import pandas as pd
def read_csv(file_path):
    """
    Load data from a CSV file into a pandas DataFrame.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file (relative or absolute)
    
    Returns:
    --------
    pd.DataFrame
        Loaded data as a pandas DataFrame
    
    Example:
    --------
    >>> data = read_csv('Iris.csv')
    """
    # pd.read_csv() reads CSV files and automatically infers data types
    return pd.read_csv(file_path)