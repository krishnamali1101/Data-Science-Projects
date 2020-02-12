from sklearn.metrics import confusion_matrix, cohen_kappa_score
from sklearn.metrics import f1_score, recall_score

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
