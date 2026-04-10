def clean_column_names(df):
    df.columns = df.columns.str.replace(r"[^\w]", "_", regex=True)
    return df