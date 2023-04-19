import json
import logging
import os
import linecache
import math
import pandas as pd
from FAQ.evaluator import Evaluator
from FAQ.bm25 import BM25Matcher
import numpy as np


matcher = BM25Matcher('./FAQ/data/question.txt', './FAQ/data/question_seg.txt', remove_stopwords=True) 
evaluator = Evaluator(topic_path="./FAQ/topic/output/model.h5", sbert_path='./FAQ/distiluse-base-multilingual-cased-v1')

def BM25_test(query, candidates, top_k=10):
	global matcher
	return matcher.match_test(query, candidates, top_k)

def mix_test(query, candidates):
	global evaluator
	indexes = BM25_test(query, candidates, 3)
	candidates = [candidates[x] for x in indexes]
	return  evaluator.getBestResponseTest(query, candidates)

def evaluator_test(query, candidates):
	global evaluator
	return  evaluator.getBestResponseTest(query, candidates)


def evaluate_recall(y, y_test, k=1):
    num_examples = float(len(y))
    num_correct = 0
    for predictions, label in zip(y, y_test):
        if label in predictions[:k]:
            num_correct += 1
    return num_correct/num_examples

if __name__ == "__main__":
	path = os.path.dirname(__file__)

	test_df = pd.read_csv("./survey/test_recall.csv")
	test_df = test_df[['Context', 'Tru_Ques', 'Dis_Ques_0', 'Dis_Ques_1', 'Dis_Ques_2', 'Dis_Ques_3', 'Dis_Ques_4', 'Dis_Ques_5', 'Dis_Ques_6', 'Dis_Ques_7', 'Dis_Ques_8']]
	y_test = np.zeros(len(test_df))


	y_bm = [BM25_test(test_df.Context[x], test_df.iloc[x,1:].values.astype('U')) for x in range(len(test_df))]
	y = [evaluator_test(test_df.Context[x], test_df.iloc[x,1:].values.astype('U')) for x in range(len(test_df))]
	# print(y)
	for n in [1, 2, 5, 10]:
		print("BM25: Recall @ ({}, 10): {:g}".format(n, evaluate_recall(y_bm, y_test, n)))
		print("PsyChatbot: Recall @ ({}, 10): {:g}".format(n, evaluate_recall(y, y_test, n)))

