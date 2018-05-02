import _pickle as cPickle

with open('y_val.pkl', 'rb') as f:
    y_val = cPickle.load(f)

with open('True_labels.txt', 'w') as f:
    for val in y_val:
        f.write(str(val))
        f.write("\n")
