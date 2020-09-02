from sklearn.metrics import confusion_matrix, cohen_kappa_score
from sklearn.metrics import f1_score, recall_score
import matplotlib.pyplot as plt
import itertools
from sklearn import metrics
import math

def PrintStats(cmat, y_test, pred):
    tpos = cmat[0][0]
    fneg = cmat[1][1]
    fpos = cmat[0][1]
    tneg = cmat[1][0]
    # calculate F!, Recall scores
    f1Score = round(f1_score(y_test, pred), 2)
    recallScore = round(recall_score(y_test, pred), 2)
    # calculate and display metrics
    print( 'Accuracy: '+ str(np.round(100*float(tpos+fneg)/float(tpos+fneg + fpos + tneg),2))+'%')
    print( 'Cohen Kappa: '+ str(np.round(cohen_kappa_score(y_test, pred),3)))
    print("Sensitivity/Recall for Model : {recall_score}".format(recall_score = recallScore))
    print("F1 Score for Model : {f1_score}".format(f1_score = f1Score))


def full_classification_report(y_train, y_test, train_pred, test_pred,
                               acc, acc_cv, time=0):

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, test_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure(figsize=(10,8))
    plot_confusion_matrix(cnf_matrix, classes=['NO','YES'], title='Confusion matrix, without normalization')

    #extracting true_positives, false_positives, true_negatives, false_negatives
    tn, fp, fn, tp = cnf_matrix.ravel()
    print("True Negatives(TN): ",tn)
    print("False Positives(FP): ",fp)
    print("False Negatives(FN): ",fn)
    print("True Positives(TP): ",tp)
    print('-'*80)

    PrintStats(cnf_matrix, y_test, test_pred)
    print('-'*80)

    print("Accuracy: %s" % acc)
    print("Accuracy CV 10-Fold: %s" % acc_cv)
    print("Running Time: %s" % datetime.timedelta(seconds=time))
    print('-'*80)
    print ("classification_report on train data:\n",metrics.classification_report(y_train, train_pred))
    print('-'*80)
    print ("classification_report on test data:\n",metrics.classification_report(y_test, test_pred))

def reg_metrics(y_test, test_pred):
    print("mean_absolute_error(MAE): ", metrics.mean_absolute_error(y_test, test_pred))
    print("root_mean_squared_error(RMSE): ", math.sqrt(metrics.mean_squared_error(y_test, test_pred)))
    print("r2_score: ", metrics.r2_score(y_test, test_pred))



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

def classification_eval(Actual_class, Pred_class, classes):
    if len(classes)==2:
        plot_roc_curve(Actual_class, Pred_class)
    
    cnf_matrix = confusion_matrix(Actual_class, Pred_class, classes)

    print("Accuracy: {}%\n".format(round(accuracy_score(Actual_class, Pred_class)*100, 2)))
    plot_confusion_matrix(cnf_matrix, classes=classes)
