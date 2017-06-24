import sys
import os
import re


def analyze_log(log_file, log_path):
    if log_file.startswith('.') or not log_file.endswith('txt'):
        return []
    print('processing {}'.format(log_file))
    with open(os.path.join(log_path,log_file), encoding='utf8') as l:
        qs = []
        for line in l:
            if 'Start:' in line:
                continue
            line = line.strip()
            # sys.stdout.write(line)
            q = re.search('(?<=\d\s\s).+(?=\s\=\=\>)', line)
            # print(q)
            if not q:
                print(line)
                continue
            q = q.group(0).lower()
            if q != 'score me' and q:
                qs.append(q)
    qs.append('')
    return qs

def analyze_question(q_file, q_path):
    if q_file.startswith('.') or not q_file.endswith('txt'):
        return []
    print('processing {}'.format(q_file))
    with open(os.path.join(q_path, q_file), encoding='utf8') as l:
        qs = []
        for line in l:
            line = line.strip().lower()
            if not qs and not line:
                continue
            if (line != 'score me' and line) or (qs[len(qs)-1] and not line):
                qs.append(line)
    qs.append('')
    return qs

if __name__ == '__main__':
    folder_name = sys.argv[1]
    total_file = os.path.join(folder_name, 'all_questions.txt')
    log_path = os.path.join(folder_name, 'logs')
    log_files = os.listdir(log_path)
    question_path = os.path.join(folder_name, 'questions')
    question_files = os.listdir(question_path)

    with open(total_file, 'w') as w:
        for f in log_files:
            x = analyze_log(f, log_path)
            for q in x:
                print(q, file=w)
        for f in question_files:
            x = analyze_question(f, question_path)
            for q in x:
                print(q, file=w)