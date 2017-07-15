from nltk.probability import FreqDist
import os
'''
frequency analysis of some output file.
'''

gold_data_file = '../data/wilkins_corrected.shuffled.51.txt'
test_preds_file = '../word_test_results.txt'
rare_label_list = '../data/rare_label_list.txt'
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
freqs = [[],[],[],[],[]]
print(frequencies.most_common())
reverse_frequencies = frequencies.most_common()[::-1]
for label, freq in reverse_frequencies:
    # if total_per_bin + freq > count_per_bin and cur_bin < 4:
    if total_per_bin > count_per_bin and cur_bin < 4:
        print('{} bin has {} counts'.format(cur_bin, total_per_bin))
        cur_bin += 1
        total_per_bin = freq
        bins[cur_bin].append(label)
        freqs[cur_bin].append(freq)
    else:
        total_per_bin += freq
        bins[cur_bin].append(label)
        freqs[cur_bin].append(freq)
print('{} bin has {} counts'.format(cur_bin, total_per_bin))
print(bins)
print([len(bin) for bin in bins])
print(freqs)

if not os.path.exists(rare_label_list):
    print(bins[0], file=open(rare_label_list, 'w'))

with open(test_preds_file) as tf:
    test_labels = []
    for line in tf:
        test_labels.append(int(line.strip()))

corrects = 0
total = 0
boolean = ['wrong', 'right']
for index, gold_label in enumerate(gold_labels):
    b = 0
    if gold_label in bins[0]:
        total += 1
        if test_labels[index] == gold_label:
            corrects += 1
            b = 1
        print('rare label prediction: index {}, gold {}, predicted {}, bool {}, test sent {!r}'.format(
        index, gold_label, test_labels[index], boolean[b], samples[index]
    ))
print('the total for this bin: {}'.format(corrects/total))