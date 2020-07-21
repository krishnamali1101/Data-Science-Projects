import pandas as pd
import string
from collections.abc import Iterable
from datetime import datetime
import os
import math

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
    
def classify(index, test_pred, classification_bins=[], classification_bins_file_path=None, 
                         default_bin_method='median', quartiles=[0.25,0.5,0.75]):
    '''
        # test_pred: 1D-list or np array
        # index: index of predictions
        # classification_bins_file_path: if bins are stored in text file, separated by new line
        # classification_bins: 
        # default_bin_method='median'/ mean/ quantile([0.25,0.5,0.75])

        Function callimng
        l = [1,2,3,4,5,6,7,8,9,20]

        filename = '/Users/gopalmali/Desktop/test.txt'

        #classify_predictions(l, l)
        # classify_predictions(l, l, classification_bins=[4,7])
        #classify_predictions(l, l, classification_bins_file_path= filename)
        # classify_predictions(l, l, default_bin_method='median')
        # classify_predictions(l, l, default_bin_method='mean')
        # classify_predictions(l, l, default_bin_method='quartile')
        # classify_predictions(l, l, default_bin_method='quartile', quartiles=[0.25, 0.75])
    '''
    
    if not classification_bins:
        try:
            ## Read classes
            with open(classification_bins_file_path) as fp:
                classification_bins = fp.read().splitlines()

            classification_bins = list(map(int, classification_bins))
        except:
            # use default classification_bins
            test_pred_series = pd.Series(test_pred)
            if default_bin_method=='median':
                classification_bins = [test_pred_series.median()]
            elif default_bin_method=='mean':
                classification_bins = [test_pred_series.mean()]
            else:
                classification_bins = list(test_pred_series.quantile(quartiles))
        
    # insert min & max in limit
    classification_bins.insert(0,-math.inf)
    classification_bins.append(math.inf)
    
    print("-- Classification Bins:", classification_bins)
    test_pred = np.array(test_pred)
    conditions = [((classification_bins[i]<=test_pred) & (test_pred<classification_bins[i+1])) for i in range(len(classification_bins)-1)]
    choices = list(range(len(conditions),0, -1))    
    test_pred_class = np.select(conditions, choices, default=max(choices))
    
    df_predict = pd.DataFrame({'CLAIM_ID':index, 'Pred':test_pred, 'Pred_class':test_pred_class})
    df_predict.set_index('CLAIM_ID',inplace=True)
    return df_predict, classification_bins
