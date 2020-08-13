import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import pylab
from scipy import stats
import pandas_profiling as pp
from random import randint
import itertools

def analyse_unique_values_in_column(df, max_unique=30, plot=False):
    get_ipython().run_line_magic('matplotlib', 'inline')

    for col in df:
        try:
            if len(df[col].unique()) < max_unique:
                #print(df[col].name, ' : ', df[col].unique())
                print(df[col].name)
                print('-'*20)
                print(value_counts(df,col).to_string())
                print('-'*80)
                print()
                #print('-'*100)

                if plot:
                    plt.figure(figsize=(10,8))
                    if len(df[col].unique())<7:
                        sns.countplot(x = col, data = df, order = df[col].value_counts().index)
                    else:
                        sns.countplot(y = col, data = df, order = df[col].value_counts().index)

                    plt.show()
                    #print('-'*80)
#                     print('-'*80)
        except Exception as e:
            print("Problem in Feature", col, e)
            print('-'*80)
        #print()


# Print Unique count in all
def print_unique_values(df, max_unique=100):
    for col in df:
        try:
            unique_values = df[col].nunique()
            print(F'{df[col].name} : {unique_values}')
            print('-'*20)

            if  unique_values < max_unique:
                print(df[col].unique())
                #print(value_counts(df,col).to_string())
                print('-'*80)
                print()
        except Exception as e:
            print("Problem in Feature", col, e)
            print('-'*80)

def missing_values_analysis(df, other_missing_values=[], figsize=(15,10)):
    for mv in other_missing_values:
        for col in df.columns:
            df[col] = np.where(df[col]==mv, np.nan, df[col])

    all_missing_values = round(df.isna().sum()*100/df.shape[0],2).sort_values(ascending=False)
    missing_values = all_missing_values[all_missing_values.values>0].sort_values(ascending=True)

    objects = missing_values.index
    y_pos = np.arange(len(objects))
    count = missing_values.values

    plt.figure(figsize=figsize)
    plt.barh(y_pos, count, align='center', alpha=0.9)
    plt.yticks(y_pos, objects)
    plt.ylabel('col names')
    plt.title('Missing values')

    plt.show()
    set_display_options()
    return all_missing_values


def segregate_columns(df):
    '''
    returns: dict of categorical_cols, bool_cols, numeric_cols, datetime_cols, other_cols
    '''
    categorical_cols = df.select_dtypes(include=['object']).columns
    bool_cols = [col_name for col_name in categorical_cols if len(df[col_name].unique())<=2]
    categorical_cols = list(set(categorical_cols)-set(bool_cols))
    bool_cols = bool_cols + list(df.select_dtypes(include=['bool']).columns)

    numeric_cols = list(df.select_dtypes(include=['float64', 'int64', 'float32']).columns)
    datetime_cols = list(df.select_dtypes(include=['datetime64','timedelta64']).columns)
    other_cols = list( set(df.columns) - set(categorical_cols+bool_cols+numeric_cols+datetime_cols))

    return {"categorical_cols":categorical_cols,
            "bool_cols":bool_cols,
            "numeric_cols":numeric_cols,
            "datetime_cols":datetime_cols,
            "other_cols":other_cols}


def outliers_analysis_using_boxplot(df, g_value = 1.5, plot=False):
    get_ipython().run_line_magic('matplotlib', 'inline')

    for col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        print('-'*80)
        print("Analysis for col: ",col)
        print("Q1: {0}\nQ3: {1}\nIQR: {2}".format(Q1, Q3, IQR))
        print("(Q1-{2}*IQR): {0}\n(Q3+{2}*IQR): {1}".format(Q1 - g_value * IQR, Q3 + g_value * IQR, g_value))

        outliers = (df[col] < (Q1 - g_value * IQR)) | (df[col] > (Q3 + g_value * IQR))
        #print(outliers)
        print("outliers: ",df[outliers][col])
        print('-'*80)

        if plot:
            plt.figure(figsize=(10,8))
            sns.set(style="whitegrid")
            ax = sns.boxplot(x=df[col])
            plt.show()

        print()

def outliers_analysis_specific_feature_using_boxplot(col_values, g_value = 1.5, plot=False):
    get_ipython().run_line_magic('matplotlib', 'inline')

    Q1 = col_values.quantile(0.25)
    Q3 = col_values.quantile(0.75)
    IQR = Q3 - Q1

    print('-'*80)
    #print("Analysis for col: ",col)
    print("Q1: {0}\nQ3: {1}\nIQR: {2}".format(Q1, Q3, IQR))
    print("(Q1-{2}*IQR): {0}\n(Q3+{2}*IQR): {1}".format(Q1 - g_value * IQR, Q3 + g_value * IQR, g_value))

    outliers = (col_values < (Q1 - g_value * IQR)) | (col_values > (Q3 + g_value * IQR))
    #print(outliers)
    print("outliers: ",col_values[outliers])
    print('-'*80)

    if plot:
        print("Before")
        plt.figure(figsize=(10,8))
        sns.set(style="whitegrid")
        ax = sns.boxplot(x=col_values)
        plt.show()

        print()

    if plot:
        print("After")
        col_values_adfter = list(set(col_values.values) - set(col_values[outliers].values))
        plt.figure(figsize=(10,8))
        sns.set(style="whitegrid")
        ax = sns.boxplot(x=col_values_adfter)
        plt.show()
        print()

    return col_values[outliers], len(col_values[outliers])*100/len(col_values)

def outliers_analysis_using_standard_deviation(df, factor = 3, plot=False):
    for col in df.columns:
        upper_lim = df[col].mean () + df[col].std () * factor
        lower_lim = df[col].mean () - df[col].std () * factor

        print('-'*80)
        print("Analysis for col: ",col)
        print("lower_lim: {0}upper_lim\n: {1}".format(lower_lim,upper_lim))

        outliers = df[(df[col] < lower_lim) | (df[col] > upper_lim)][col]
        print("outliers: ",outliers.values)
        print('-'*80)

        if plot:
            pylab.rcParams['figure.figsize'] = (10.0, 8.0)
            ax = plt.subplot(111)
            x = df[(df[col] > lower_lim) & (df[col] < upper_lim)][col] # data other thn outliers

            x = np.sort(x)
            #ax.set_xlim([-2.5,2.5])
            ax.plot(x,stats.norm.pdf(x))
            plt.show()
#             plt.figure(figsize=(10,8))
#             sns.set(style="whitegrid")
#             ax = sns.boxplot(x=df[col])
#             plt.show()

        print()

#df = df[~outliers]

def profile_report(df, out_file='profile_report.html', dump_profile_report=True):
    pfr = pp.ProfileReport(df)

    if dump_profile_report:
        pfr.to_file(out_file)

    pfr.to_notebook_iframe()


def crosstab(col1,col2):
    df = pd.crosstab(col1, col2, margins=True)

    df.plot(kind='bar',figsize=(15,10))
    return df

# looking at any random row
def print_random_row(df):
    set_display_options()
    randi = randint(0,df.shape[0]-1)
    print("Row No: ", randi)
    return df.iloc[randi]

def formated_print(values, num_of_cols=4, col_width=30):
    ''' values: list, tuple, set
    '''
    print(len(values))
    for i,col in enumerate(values):
        if i%num_of_cols==0:
            print()
        print('{:{}}'.format(col,col_width), end='')

def set_display_options(max_rows=500, max_columns=500, width=1000, max_colwidth=-1):
    '''set_display_options for jupyter notebook'''
    pd.options.display.max_rows
    pd.set_option('display.max_rows', max_rows)
    pd.set_option('display.max_columns', max_columns)
    pd.set_option('display.width', width)
    pd.set_option('display.max_colwidth', max_colwidth)


def find_columns_to_del(df, critical_missing_value_percentage=0.9, id_col_unique_values_percentage=0.9):
    # Features with single constant values to remove.
    nunique = df.apply(pd.Series.nunique)
    single_constant_values_cols = list(nunique[nunique<=1].index)

    # Features with more than 90% of the data missing to remove.
    critical_missing_value_cols = list(df[df.columns[
        df.isnull().mean()>=critical_missing_value_percentage]].columns)

    # find ID columns
    id_columns = [col for col in df.columns
                  if (len(df[col].unique())/df.shape[0])>=id_col_unique_values_percentage]

    return {"single_constant_values_cols":single_constant_values_cols,
            "critical_missing_value_cols":critical_missing_value_cols,
            "id_columns":id_columns}


# print some full rows(0 nan value rows)
def print_full_rows(df, how_many=1):
    '''
    Try this code to understand this function
    df1 = pd.DataFrame({'A':[None,None,None,2,3],'B':[1,None,3,None,4]})
    print_full_rows(df1)
    '''
    i =0
    for row in range(df.shape[0]):
        if i==how_many:
            break
        b=True
        for col in df.columns:
            #print(df[col].isna())
            b = b and ~df[col].isna()[row]

        if b:
            i+=1
            print(df.iloc[row])


    # better logic(but will print all full rows)
    print(df1[df1.isnull().mean(axis=1)==1])

def plot_corr_mat(df):
    plt.figure(figsize=(20,15))

    sns.heatmap(df.corr(),
                xticklabels=df.corr().columns.values,
                yticklabels=df.corr().columns.values,
                annot=True);

    print(df.corr())

def value_counts(df,col, round_val=2):
    return pd.DataFrame(list(zip(df[col].value_counts().index,
                                    df[col].value_counts().values,
                                    df[col].value_counts(normalize=True).mul(100).round(round_val).values)),
                                columns=[col,"Count", "%Count"])

def df_to_formatted_str(df, headers=True, join_with=' -- '):
    l1 = []
    if headers:
        l1 = [str(list(df.columns))]

    for i in range(len(df)):
        l2 = []
        for col in df.columns:
            l2.append(df.loc[i,col])
        l1.append(str(tuple(l2)))
    return join_with.join(l1)

# Function which returns subset or r length from n
def combinations_single_list(l):
    list_all_combinations = []
    for r in range(1, len(l)+1):
        list_all_combinations.extend(list(itertools.combinations(l, r)))

    return list_all_combinations


def combinations_multi_list(list_of_lists):
    return list(itertools.product(*list_of_lists))



def data_distribution(df, other_missing_values=[], max_nunique=10, plot=False, figsize=(15,20), 
                      is_Unique_values_distribution = True, is_print_random_row=True,
                       is_find_columns_to_del = True, is_segregate_columns=True,
                      critical_missing_value_percentage=0.9, id_col_unique_values_percentage=0.9):
    '''Features: Shape, missing values, unique values, data distribution, segregate_columns, 
    find_columns_to_del, print_random_row'''

    print("Shape: ", df.shape)
    print('='*80)
    print()
    
    for mv in other_missing_values:
        for col in df.columns:
            df[col] = np.where(df[col]==mv, np.nan, df[col])

    all_missing_values = df.isna().sum().sort_values(ascending=False)
    #all_missing_values = round(df.isna().sum()*100/df.shape[0],2).sort_values(ascending=False)
    all_missing_values_per = round(all_missing_values*100/df.shape[0],4)
    missing_values = all_missing_values_per[all_missing_values_per.values>0].sort_values(ascending=True)

    if plot:
        objects = missing_values.index
        y_pos = np.arange(len(objects))
        count = missing_values.values

        plt.figure(figsize=figsize)
        plt.barh(y_pos, count, align='center', alpha=0.9)
        plt.yticks(y_pos, objects)
        plt.ylabel('col names')
        plt.title('Missing values')

        plt.show()
        set_display_options()
        print('='*80)
        print()
    # create df
    all_missing_values_df = pd.DataFrame()
    all_missing_values_df['Feature'] = all_missing_values.index
    all_missing_values_df['Missing_values'] = all_missing_values.values
    all_missing_values_df['% Missing_values'] = all_missing_values_per.values
    all_missing_values_df['available_values'] = df.shape[0] - all_missing_values.values
    all_missing_values_df['Unique_values_count'] = df[all_missing_values.index].nunique().values
    
    if is_Unique_values_distribution:
        all_missing_values_df['Unique_values_distribution'] = [df_to_formatted_str(value_counts(df,col), headers=False, join_with=' -- ') 
         if df[col].nunique()<=max_nunique else '' for col in all_missing_values_df.Feature]
    
    if is_segregate_columns:
        print("Diffrent columns available: ")
        segregated_columns_dict = segregate_columns(df)
        for k,v in segregated_columns_dict.items():
            print(k," : ",v)
            print('-'*50)
        print('='*80)
        print()
    
    if is_find_columns_to_del:
        print("Suggested columns to delete: ")
        cols_to_del_dict = find_columns_to_del(df, critical_missing_value_percentage, id_col_unique_values_percentage)
        for k,v in cols_to_del_dict.items():
            print(k," : ",v)
            print('-'*50)
        print('='*80)
        print()
        
    if is_print_random_row:
        print("Random Row:\n",print_random_row(df))
        print('='*80)
        print()
    
    return all_missing_values_df.set_index('Feature')
