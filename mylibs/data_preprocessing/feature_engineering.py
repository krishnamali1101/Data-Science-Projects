def apply_sampling(X_train, y_train, target_column_name, sampling_type='SMOTETomek'):
    # apply sampling on training Data set
    from collections import Counter

    print('Original dataset shape %s' % Counter(y_train))

    if sampling_type=='RandomUnderSampler':
        from imblearn.under_sampling import RandomUnderSampler
        sampler = RandomUnderSampler(sampling_strategy=1, random_state=23)
        X_train, y_train = sampler.fit_resample(X_train, y_train)
    elif sampling_type=='SMOTETomek':
        from imblearn.combine import SMOTETomek
        sampler = SMOTETomek(random_state=42)
        X_train, y_train = sampler.fit_resample(X_train, y_train)
    else:
        # No sampling
        pass

    print('Resampled dataset shape %s' % Counter(y_train))
    print(X_train.shape, y_train.shape)

    return X_train, y_train


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

# Target var should be numeric and feature_to_encode should be categorical 
def LabelEncoder_groupby_label_sortedby_meanof_targetvar(df, feature_to_encode, target_var):
    from pandas.api.types import is_string_dtype, is_numeric_dtype
    if not (is_numeric_dtype(df[target_var]) & is_string_dtype(df[feature_to_encode])):
        print("ErrorMsg: Target var should be numeric and feature_to_encode should be categorical")
        return {}
    
    temp = df.groupby(feature_to_encode)[target_var].mean().sort_values()

    res = list(zip(*enumerate(temp.index)))
    return dict(zip(res[1],res[0]))

# VIF_test for feature selection/remove
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
