def convert_date_to_datetime(date):
    if not pd.isna(date):
        d, m, y = str(date).split('-')
        dt = "-".join([d, m, '%d%s' % (19 if int(y) >= 25 else 20, y)])
        return pd.to_datetime(dt, format='%d-%b-%Y')
    else:
        return date


def convert_date_to_str(date):
    import pandas as pd
    if not pd.isna(date):
        try:
            d, m, y = str(date).split('-')
            return "-".join([d, m, '%d%s' % (19 if int(y) >= 25 else 20, y)])
        except:
            return date
    else:
        return date


# remove_punctuation from col
def filter_df(df, skip_cols):
    import string
    def filter_column(col_data):
        from collections.abc import Iterable

        if isinstance(col_data, Iterable):
            col_data = col_data.lower().strip().replace(' ', '_')
            col_data = ''.join([i for i in col_data if i not in frozenset(string.punctuation.replace('_',''))])
        return col_data

    #filter col_names
    # df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
    #df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

    #filter col_values
    df_categorical = df.select_dtypes(include=object)
    cat_columns = list(set(df_categorical.columns) - set(skip_cols))
    for col in cat_columns:
        df[col] = df[col].apply(filter_column)
    return df
