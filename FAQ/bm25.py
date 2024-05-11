import math
import heapq
import os
import logging
from .matcher import Matcher


class BM25Matcher(Matcher):
	def __init__(self, input_path, output_path, remove_stopwords):
		super().__init__()
		self.remove_sw = remove_stopwords
		self.total = 0
		self.wordset = set()
		self.word_location_record = dict()
		self.word_idf = dict()
		self.f = []
		self.df = {}
		self.idf = {}
		self.k1 = 1.5
		self.b = 0.75
		self.searcher = QuickSearcher()
		if "question" in input_path:
			self.type = "question"
		elif "answer" in input_path:
			self.type = "answer"
		else:
			self.type = "sentence"

		if remove_stopwords:
			# self.loadStopwords("./FAQ/data/stopwords/chinese_sw.txt")
			# self.loadStopwords("./FAQ/data/stopwords/specialMarks.txt")
			self.loadStopwords("data/stopwords/chinese_sw.txt")
			self.loadStopwords("data/stopwords/specialMarks.txt")

		self.segmentData(input_path, output_path, remove_stopwords, self.type)
		self.initBM25()
		self.searcher.buildInvertedIndex(self.seg_data)

	def initBM25(self):
		print("Initializing BM25 module for %s" % self.type)

		self.total = len(self.seg_data)
		self.avgdl = sum([len(item) + 0.0 for item in self.seg_data]) / self.total

		for item in self.seg_data:
			tmp = {}
			for word in item:
				if not word in tmp:
					tmp[word] = 0
				tmp[word] += 1
			self.f.append(tmp)
			for k, v in tmp.items():
				if k not in self.df:
					self.df[k] = 0
				self.df[k] += 1
		for k, v in self.df.items():
			self.idf[k] = math.log(self.total - v + 0.5) - math.log(v + 0.5)

		print("BM25 module has been initialized for %s." % self.type)

	def calculateSim(self, query, index):
		score = 0
		for word in query:
			if word not in self.f[index]:
				continue
			d = len(self.seg_data[index])
			score += (self.idf[word] * self.f[index][word] * (self.k1 + 1)
					/ (self.f[index][word] + self.k1 * (1 - self.b + self.b * d / self.avgdl)))

		return score

	def calculateIDF(self):
		if len(self.wordset) == 0:
			self.buildWordSet()
		if len(self.word_location_record == 0):
			self.buildWordLocationRecord()
		for word in self.wordset:
			self.word_idf[word] = math.log2((self.D + 0.5) / (self.word_location_record[word] + 0.5))

	def buildWordSet(self):
		for item in self.seg_data:
			for word in item:
				self.wordset.add(word)

	def buildWordLocationRecord(self):
		for index, item in enumerate(self.seg_data):
			for word in item:
				if self.word_location_record[word] is None:
					self.word_location_record[word] = set()
				self.word_location_record[word].add(index)

	def match(self, query, top_k=15):
		seg_query = self.segment_word(query, self.remove_sw)

		indexes = list(self.searcher.quickSearch(seg_query))
		scores = [self.calculateSim(seg_query, i) for i in indexes]
		if len(indexes) > top_k:
			li = heapq.nlargest(top_k, range(len(indexes)), scores.__getitem__)
			return [indexes[i] for i in li], [scores[i] for i in li]
		else:
			return indexes, scores
			
	def match_test(self, query, candidates, top_k):
		query = query.split(' ')
		scores = []
		for i in candidates:
			score = 0
			words = i.split(' ')
			d = len(words)
			for q in query:
				if q not in words:
					continue
				freq = words.count(q)
				score += (self.idf[q] * freq * (self.k1 + 1)
						/ (freq + self.k1 * (1 - self.b + self.b * d / self.avgdl)))	
			scores.append(score)
		
		li = heapq.nlargest(top_k, range(len(candidates)), scores.__getitem__)
		return li


class QuickSearcher:
	'''Inverted index'''
	def __init__(self, docs=None):
		self.inverted_word_dict = dict()

	def buildInvertedIndex(self, docs):
		for doc_id, doc in enumerate(docs):
			for word in doc:
				if word not in self.inverted_word_dict.keys():
					self.inverted_word_dict[word] = set()
				self.inverted_word_dict[word].add(doc_id)

	def quickSearch(self, query):
		result = set()
		for word in query:
			if word in self.inverted_word_dict.keys():
				result = result.union(self.inverted_word_dict[word])

		return result