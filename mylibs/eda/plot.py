def display_plot(df, col_to_exclude='', object_mode = True):
    import matplotlib.pyplot as plt
    import seaborn as sns
    n = 0
    this = []
    if object_mode:
        nrows = 4
        ncols = 4
        width = 20
        height = 20
    else:
        nrows = 2
        ncols = 2
        width = 14
        height = 10

    for column in df.columns:
        if object_mode:
            if (df[column].dtypes == 'O') & (column != col_to_exclude):
                this.append(column)
        else:
            if (df[column].dtypes != 'O'):
                this.append(column)

    fig, ax = plt.subplots(nrows, ncols, sharex=False, sharey=False, figsize=(width, height))
    for row in range(nrows):
        for col in range(ncols):
            if object_mode:
                g = sns.countplot(df[this[n]], ax=ax[row][col])
            else:
                g = sns.distplot(df[this[n]], ax = ax[row][col])
            ax[row,col].set_title("Column name: {}".format(this[n]))
            ax[row, col].set_xlabel("")
            ax[row, col].set_ylabel("")
            n += 1
    plt.show();
    return None

def donut_chart(col_values, col_name='', normalize=True, figsize=(17,8)):
    import matplotlib.pyplot as plt
    get_ipython().run_line_magic('matplotlib', 'inline')
    import seaborn as sns

    # The slices will be ordered and plotted counter-clockwise.
    value_counts_dict = col_values.value_counts(normalize)

    new_value_counts_dict = {}
    for index,value in zip(value_counts_dict.index, value_counts_dict.values):
        if value > 0.01:
            new_value_counts_dict[index] = value
        elif 'other_values' in new_value_counts_dict.keys():
            new_value_counts_dict['other_values'] += value
        else:
            new_value_counts_dict['other_values'] = value

    labels = new_value_counts_dict.keys()
    sizes = new_value_counts_dict.values()
    #explode = (0 for i in range(len(labels)))

    plt.figure(figsize=figsize
    plt.title(col_name)
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True)

    #draw a circle at the center of pie to make it look like a donut
    centre_circle = plt.Circle((0,0),0.70,color='black', fc='white',linewidth=1.25)
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    # Set aspect ratio to be equal so that pie is drawn as a circle.
    plt.axis('equal')
    plt.show()

    print(col_name,"\n",col_values.value_counts())
    print('-'*80)
