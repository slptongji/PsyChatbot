import os
import random
import numpy as np
from _tkinter import _flatten


if __name__ == '__main__':
    label_file = open('label.txt', 'r', encoding='utf-8')
    question_file = open('question.txt', 'r', encoding='utf-8')
    topic_type = ["人际",     "家庭",     "恋爱婚姻",     "情绪",     "成长",     "治疗",     "职业" ]
    data = [[] for i in range(len(topic_type))]
    for label, question in zip(label_file, question_file):
        label = label.strip().replace(" ", "")
        question = question.strip().replace(" ", "")
        if not label or not question:
            continue

        for i, topic in enumerate(topic_type):
            if topic in label:
                data[i].append(label + ' ' + question)
    label_file.close()
    question_file.close()

    lens = [len(i) for i in data]
    print(lens)
    min_len = min(lens)
    print(min_len)
    data = [i[:min_len] for i in data]
    data = list(_flatten(data))
    print(np.array(data).shape)
    random.shuffle(data)
    train_num = int(len(data) * 0.9)
    train_set = data[:train_num]
    test_set = data[train_num:]
    print(len(train_set))
    print(len(test_set))

    with open('train.txt', 'w', encoding='utf-8') as f:
        for i in train_set:
            f.write(i + '\n')

    with open('test.txt', 'w', encoding='utf-8') as f:
        for i in test_set:
            f.write(i + '\n')
