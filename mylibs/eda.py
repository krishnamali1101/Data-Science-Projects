def analyse_unique_values_in_column(df, max_unique=30, plot=False):
    import matplotlib.pyplot as plt
    get_ipython().run_line_magic('matplotlib', 'inline')
    import seaborn as sns

    for col in df:
        if len(df[col].unique()) < max_unique:
            print(df[col].name, ' : ', df[col].unique())
            print('-'*80)
            #print('-'*100)

            if plot:
                plt.figure(figsize=(10,8))
                if len(df[col].unique())<7:
                    sns.countplot(x = col, data = df, order = df[col].value_counts().index)
                else:
                    sns.countplot(y = col, data = df, order = df[col].value_counts().index)

                plt.show()
                print('-'*80)
                print('-'*80)
            print()


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




def missing_values_analysis(df):
    all_missing_values = round(df.isna().sum()*100/df.shape[0],2).sort_values(ascending=False)
    missing_values = all_missing_values[all_missing_values.values>0].sort_values(ascending=True)

    import numpy as np
    import matplotlib.pyplot as plt

    objects = missing_values.index
    y_pos = np.arange(len(objects))
    count = missing_values.values

    plt.figure(figsize=(15,10))
    plt.barh(y_pos, count, align='center', alpha=0.9)
    plt.yticks(y_pos, objects)
    plt.ylabel('col names')
    plt.title('Missing values')

    plt.show()

    return all_missing_values


# In[5]:


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


# In[6]:


def segregate_columns(df):

    categorical_cols = df.select_dtypes(include=['object']).columns

    bool_cols = [col_name for col_name in categorical_cols if len(df[col_name].unique())<=2]

    categorical_cols = list(set(categorical_cols)-set(bool_cols))

    bool_cols = bool_cols + list(df.select_dtypes(include=['bool']).columns)

    numeric_cols = list(df.select_dtypes(include=['float64', 'int64']).columns)

    datetime_cols = list(df.select_dtypes(include=['datetime64','timedelta64']).columns)

    other_cols = list( set(df.columns) - set(categorical_cols+bool_cols+numeric_cols+datetime_cols))

    return categorical_cols, bool_cols, numeric_cols, datetime_cols, other_cols


# In[7]:


def outliers_analysis_using_boxplot(df, g_value = 1.5, plot=False):

    import seaborn as sns
    import matplotlib.pyplot as plt
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

    import seaborn as sns
    import matplotlib.pyplot as plt
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
    if not EDA:
        return

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
            import matplotlib.pyplot as plt
            import pylab
            import numpy as np
            from scipy import stats

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
# In[8]:



def feature_scaling(df, col_name_list, replace=False,drop=True):
    from sklearn.preprocessing import StandardScaler

    for col in col_name_list:
        if replace:
            df[col] = StandardScaler().fit_transform(df[col].values.reshape(-1,1))
        else:
            df['{}_normalized'.format(col)] = StandardScaler().fit_transform(df[col].values.reshape(-1,1))
            if drop:
                df = df.drop([col],axis=1)

    return df


def reg_metrics(y_test, test_pred):
    from sklearn import metrics
    import math

    print("mean_absolute_error(MAE): ", metrics.mean_absolute_error(y_test, test_pred))
    print("root_mean_squared_error(RMSE): ", math.sqrt(metrics.mean_squared_error(y_test, test_pred)))
    print("r2_score: ", metrics.r2_score(y_test, test_pred))


# In[9]:


def VIF_test(df):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import pandas as pd

    cols_list = []
    vif_list = []
    for i in range(df.shape[1]):
        cols_list.append(df.columns[i])
        vif_list.append(variance_inflation_factor(df.values, i))

    vif_df = pd.DataFrame({'Feature':cols_list, 'VIF': vif_list}).sort_values(by='VIF', ascending=False)
    return vif_df


def profile_report(df, out_file='profile_report.html', dump_profile_report=True):
    if not EDA:
        return

    try:
        import pandas_profiling as pp
    except:
        #pip install pandas-profiling
        import pandas_profiling as pp

    pfr = pp.ProfileReport(df)

    if dump_profile_report:
        pfr.to_file(out_file)

    pfr.to_notebook_iframe()


# looking at any random row
def print_random_row(df):
    from random import randint
    randi = randint(0,df.shape[0]-1)

    print("Row No: ", randi)
    return df.iloc[randi]
