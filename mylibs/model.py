
# Function that runs the requested algorithm and returns the accuracy metrics
def fit_ml_algo(algo, X_train, y_train, X_test, cv, probs_threshold=.4):
    # One Pass
    model = algo.fit(X_train, y_train)
    probs = model.predict(X_test)
    #print(probs)
    test_pred = np.where(probs >= probs_threshold, 1, 0)
    #print(test_pred)

    if (isinstance(algo, (LogisticRegression,
                          KNeighborsClassifier,
                          GaussianNB,
                          DecisionTreeClassifier,
                          RandomForestClassifier,
                          GradientBoostingClassifier))):
        probs = model.predict_proba(X_test)[:,1]
        test_pred = np.where(probs >= probs_threshold, 1, 0)

#     else:
#         probs = "Not Available"
    acc = round(model.score(X_test, y_test) * 100, 2)
    # CV
    train_pred = model_selection.cross_val_predict(algo,
                                                  X_train,
                                                  y_train,
                                                  cv=cv,
                                                  n_jobs = -1)
    acc_cv = round(metrics.accuracy_score(y_train, train_pred) * 100, 2)
    return train_pred, test_pred, acc, acc_cv, probs



# Utility function to report best scores
def report(results, n_top=5):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")




# Function that runs the requested algorithm and returns the accuracy metrics
def fit_ml_algo_lgbm(algo, X_train, y_train, X_test, cv, probs_threshold=.4):

    # fit model(for acc)
    model = algo.fit(X_train, y_train)
    #acc = round(model.score(X_test, y_test) * 100, 2)



    # Model building and training
    import lightgbm as lgb
    train_data = lgb.Dataset(X_train,#label=y_train.INCURRED_TOTAL,
                             label=y_train,
                            # categorical_feature = cat_lgb
                            ) ##only when label encoding. (or target +label)
    params = {'num_leaves':60, 'objective':'binary','max_depth':6,'learning_rate':0.1,'max_bin':2000,"is_unbalance":True,
            #"categorical_feature":[1,2,3,4,5,6,8,10,11,13,14,15,16,17,18],
             'boost_from_average':True,'reg_sqrt':True,
             "boosting":"gbdt",
             "use_missing":True,
             "zero_as_missing":True,
             #"max_cat_to_onehot":10,
             "top_k":10000,
             #"cat_smooth":10.0,"cat_l2":10
            "tree_learner":"data",
             "min_child_weight":1.1,
             'sparse_threshold':0.8,
             'is_enable_sparse':True,
             'cat_smooth':10,
             'cat_l2':10
            }
    params['metric'] = ['auc', 'binary_logloss']
    #param['metric'] = ['rmse',"rmsle",'l1','l2']
    #training our model using light gbm
    # num_round=20000


    # Train
    clf = lgb.train(params, train_data, num_boost_round=100)

    # #Prediction
    probs=clf.predict(X_test)

    #convert into binary values
    test_pred = np.where(probs >= probs_threshold, 1, 0)

    #calculating accuracy
    acc = accuracy_score(test_pred,y_test)*100

    # CV
    train_pred = model_selection.cross_val_predict(algo,
                                                  X_train,
                                                  y_train,
                                                  cv=cv,
                                                  n_jobs = -1)
    acc_cv = round(metrics.accuracy_score(y_train, train_pred) * 100, 2)
    return train_pred, test_pred, acc, acc_cv, probs

# Importing the Keras libraries and packages
def fit_ANN(X_train, y_train, input_dim):
    import keras
    from keras.models import Sequential
    from keras.layers import Dense

    # Seed Random Number to get constant result
    from numpy.random import seed
    seed(1)

    # Initialising the ANN
    classifier = Sequential()
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(units =15 , kernel_initializer = 'uniform', activation = 'relu', input_dim = input_dim))
    # Adding the second hidden layer
    classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu'))
    # Adding the third hidden layer
    classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu'))
    # Adding the output layer
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    # Compiling the ANN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    # Fitting the ANN to the Training set
    classifier.fit(X_train, y_train, batch_size = 32, epochs = 100)
    return classifier


def stepwise_selection(X, y,
                       initial_list=[],
                       threshold_in=0.01,
                       threshold_out = 0.05,
                       verbose=True):
    import statsmodels.api as sm
    import pandas as pd
    """ Perform a forward-backward feature selection
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.argmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included
