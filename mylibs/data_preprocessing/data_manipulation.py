import pandas as pd
import string
from collections.abc import Iterable
from datetime import datetime
import os

def convert_date_to_datetime(date):
    if not pd.isna(date):
        d, m, y = str(date).split('-')
        dt = "-".join([d, m, '%d%s' % (19 if int(y) >= 25 else 20, y)])
        return pd.to_datetime(dt, format='%d-%b-%Y')
    else:
        return date


def convert_date_to_str(date):
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
    def filter_column(col_data):
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

def find_nth(string, substring, n):
    if (n == 1):
        return string.find(substring)
    else:
        return string.find(substring, find_nth(string, substring, n - 1) + 1)

# extract code between square brackets
# loc is loc of first code
def extract_code(s, start_with='[', ends_with=']', loc=1):
    try:
        if isinstance(s, str):
            return s[find_nth(s,start_with,loc)+1:find_nth(s,ends_with,loc)]
        else:
            return s
    except:
        print("Problem in ",s)
        return None

# save file (csv)
def save_df(df, filename, path):
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    file_name = path+ os.path.sep+"{0}_{1}.csv".format(filename, datetime_str)
    df.to_csv(file_name,index=False)
    print('-'*80)
    print("File saved...", file_name)
    print('-'*80)

def read_excel(filename):
    xls = pd.ExcelFile(filename)
    print("Available Sheets: ",xls.sheet_names)

    df_dict = {}
    for sheet in xls.sheet_names:
        df_dict[sheet] = pd.read_excel(filename, sheet_name=sheet, index_col=0)
    return df_dict


def write_excel(df_dict,filename, engine='xlsxwriter'):
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(filename, engine=engine)

    for col in df_dict.keys():
        df_dict[col].to_excel(writer, sheet_name=col)

    writer.save()
