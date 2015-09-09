import numpy as np


ratio = 40
idn = '925545'
train_label=np.loadtxt('usps_blankout' + str(ratio) + '_train.label');
test_label=np.loadtxt('usps_blankout' + str(ratio) + '_test.label');

train_label_o=np.loadtxt('usps_blankout' + str(ratio) + '_train.d2s_' + idn + '.label_o');
test_label_o=np.loadtxt('usps_blankout' + str(ratio) + '_test.d2s_' + idn + '.label_o');

k=int(np.max(train_label_o) + 1)

center_label = np.zeros(k)
for j in range(0, k):
    hist,bin_edges = np.histogram([train_label[i] for i in range(0, 8000) if train_label_o[i] == j], range(0,11))
    center_label[j]=np.argmax(hist)

predict_label = [center_label[test_label_o[i]] for i in range(0, 3000)]


error = sum([1 for i in range(0,3000) if not predict_label[i] == test_label[i]]) / float(3000)

print k,error
