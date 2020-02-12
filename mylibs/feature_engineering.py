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
