from nltk.probability import FreqDist

'''
frequency analysis of some output file.
'''

gold_data_file = '../data/wilkins_corrected.shuffled.51.txt'
test_preds_file = '../word_test_results.txt'
verbose = 0

frequencies = FreqDist()
samples = []

gold_labels = []
for line in open(gold_data_file).readlines():
    label = int(line.strip().split('\t')[0])
    gold_labels.append(label)
    samples.append(line.strip().split('\t')[1])
    frequencies[label] += 1

bins = [[],[],[],[],[]]
count_per_bin = 4330 / 5
total_per_bin = 0
cur_bin = 0
print(frequencies.most_common())
for label, freq in frequencies.most_common():
    if total_per_bin + freq > count_per_bin and cur_bin < 4:
        print('{} bin has {} counts'.format(cur_bin, total_per_bin))
        cur_bin += 1
        total_per_bin = freq
        bins[cur_bin].append(label)
    else:
        total_per_bin += freq
        bins[cur_bin].append(label)

print(bins)

with open(test_preds_file) as tf:
    test_labels = []
    for line in tf:
        test_labels.append(int(line.strip()))

corrects = 0
total = 0
boolean = ['wrong', 'right']
for index, gold_label in enumerate(gold_labels):
    b = 0
    if gold_label in bins[-1]:
        total += 1
        if test_labels[index] == gold_label:
            corrects += 1
            b = 1
        print('rare label prediction: index {}, gold {}, predicted {}, bool {}, test sent {!r}'.format(
        index, gold_label, test_labels[index], boolean[b], samples[index]
    ))
print('the total for this bin: {}'.format(corrects/total))