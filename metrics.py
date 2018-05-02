from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score

from sklearn.metrics import precision_recall_fscore_support


predicted_labels = []
true_labels = []

f = open("Predictions_100d.txt","r")
for i in f:
    i = i.strip('[]')
    i = i.strip('\n')
    i = i.strip(']')
    i = i[1:]
    prob = i.split(' ')
    ls = []
    for val in prob:
        if val != '':
            ls.append(val)
    val1 = float(ls[0])
    val2 = float(ls[1])
    if val1 > val2:
        predicted_labels.append(0)
    else:
        predicted_labels.append(1)

f = open("True_labels.txt","r")
for i in f:
    i = i.strip('[]')
    i = i.strip('\n')
    i = i.strip(']')
    i = i[1:]
    prob = i.split(' ')
    ls = []
    for val in prob:
        if val != '':
            ls.append(val)
    val1 = float(ls[0])
    val2 = float(ls[1])
    if val1 > val2:
        true_labels.append(0)
    else:
        true_labels.append(1)


print(precision_recall_fscore_support(true_labels,predicted_labels,average=None))
print
print
print('Accuracy:', accuracy_score(true_labels, predicted_labels))
print('F1 score:', f1_score(true_labels, predicted_labels))
print('Recall:', recall_score(true_labels, predicted_labels))
print('Precision:', precision_score(true_labels, predicted_labels))
print('\n clasification report:\n', classification_report(true_labels,predicted_labels))
print('\n confussion matrix:\n',confusion_matrix(true_labels, predicted_labels))
