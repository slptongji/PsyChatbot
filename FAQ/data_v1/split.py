import os
import numpy as np
import jieba
import pandas as pd

def segment_word(sentence, stopwords):
    words = [word for word in jieba.cut(sentence)]
    if stopwords:
        rm_words = [word for word in words if word not in stopwords]
        return ' '.join(rm_words)
    return ' '.join(words)



if __name__ == '__main__':
    answer_file  = open('./best_answer.txt', 'r', encoding='utf-8')
    question_file = open('./question.txt', 'r', encoding='utf-8')

    # load stopwords
    stopwords = set()
    with open('./stopwords/chinese_sw.txt', 'r', encoding='utf-8') as sw:
        for word in sw:
                stopwords.add(word.strip('\n'))
    with open('./stopwords/specialMarks.txt', 'r', encoding='utf-8') as sw:
        for word in sw:
                stopwords.add(word.strip('\n'))

    # construct dataset
    pos_df = pd.DataFrame(columns=['Context', 'Utterance', 'Label'])
    neg_df = pd.DataFrame(columns=['Context', 'Utterance', 'Label'])
    for ques, ans in zip(question_file, answer_file):
        ques = segment_word(ques, stopwords=stopwords)
        ans = segment_word(ans, stopwords=stopwords)
        pos_df = pos_df.append({'Context':ques, 'Utterance':ans, 'Label':1}, ignore_index=True)
        if len(pos_df) % 5000 == 0:
            print('Appended data: {}'.format(len(pos_df)))
    for idx, row in pos_df.iterrows():
        ques = row['Context']
        ran_row = pos_df.sample(n=1, random_state=None)
        while idx == ran_row.index:
            ran_row = pos_df.sample(n=1, random_state=None)
            print("Cyciling...")
        ans = ran_row['Utterance']
        neg_df = neg_df.append({'Context':ques, 'Utterance':ans, 'Label':0}, ignore_index=True)
        if len(neg_df) % 5000 == 0:
            print('Appended data: {}'.format(len(neg_df)))
    answer_file.close()
    question_file.close()

    pos_train = pos_df.sample(frac=0.8, random_state=2)
    neg_train = neg_df.sample(frac=0.8, random_state=2)
    pos_val = pos_df[~pos_df.index.isin(pos_train.index)]
    neg_val = neg_df[~neg_df.index.isin(neg_train.index)]
    train = pd.concat([pos_train, neg_train])
    val = pd.concat([pos_val, neg_val])
    train = train.sample(frac=1).reset_index(drop=True)
    val = val.sample(frac=1).reset_index(drop=True)

    print(len(train), "train +", len(val), "val")
    train.to_csv('train.csv')
    val.to_csv('val.csv')