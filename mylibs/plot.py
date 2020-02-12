def display_plot(df, col_to_exclude='', object_mode = True):
    if not EDA:
        return

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


def crosstab(col1,col2):
    if not EDA:
        return
    import pandas as pd
    df = pd.crosstab(col1, col2, margins=True)

    df.plot(kind='bar',figsize=(15,10))
    return df


def donut_chart(col_values, col_name='', normalize=True):
    if not EDA:
        return

    import matplotlib.pyplot as plt
    %matplotlib inline
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

    plt.figure(figsize=(17,8))
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


# calculate the fpr and tpr for all thresholds of the classification
def plot_roc_curve(y_test, preds):
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


#Evaluation of Model - Confusion Matrix Plot
import itertools

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.tight_layout()
    plt.show()


def plot_roc_curves(y_test, prob, model):
    fpr, tpr, threshold = metrics.roc_curve(y_test, prob)
    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, 'b', label = model + ' AUC = %0.2f' % roc_auc, color=np.random.rand(3,))
    plt.legend(loc = 'lower right')
    return roc_auc*100
