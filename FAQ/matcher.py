import logging
import os
import jieba

class Matcher:
	def __init__(self):
		logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
		self.seg_data = []
		self.stopwords = set()

	def loadStopwords(self, path):
		with open(path, 'r', encoding='utf-8') as sw:
			for word in sw:
				self.stopwords.add(word.strip('\n'))

	# def segmentWord(self, sentence):
	# 	return [word for word in jieba.cut(sentence)]

	def segment_word(self, sentence, stopwords):
		words = [word for word in jieba.cut(sentence)]
		if stopwords:
			rm_words = [word for word in words if word not in self.stopwords]
			return rm_words
		return words

	def segmentData(self, input_path, output_path, remove_stopwords, seg_type='question'):
		logging.info("Loading all the %ss..." % seg_type)
		with open(input_path, 'r', encoding='utf-8') as data:
			dataset = [line.strip('\n').strip('\t') for line in data]

		logging.info("Segmenting all the %ss..." % seg_type)
		count = 0

		if not os.path.exists(output_path):

			self.seg_data = []
			for item in dataset:
				# if remove_stopwords:
				# 	rm_word = [word for word in self.segmentWord(item)
				# 				if word not in self.stopwords]
				# 	self.seg_data.append(rm_word)
				# else:
				# 	self.seg_data.append(self.segmentWord(item))
				self.seg_data.append(self.segment_word(sentence=item, stopwords=remove_stopwords))

				count += 1
				if count % 1000 == 0:
					logging.info("Segmented %ss: %d" % (seg_type, count))

			with open(output_path, 'w', encoding='utf-8') as seg_file:
				for item in self.seg_data:
					seg_file.write(' '.join(item) + '\n')

			logging.info("All the %ss have been segmented and the results are stored in %s." % (seg_type, output_path))

		else:
			logging.info("Reading existing segmented results of %s..." % seg_type )

			with open(output_path, 'r', encoding='utf-8') as seg_file:
				for line in seg_file:
					line = line.strip('\n')
					seg = line.split()

					if remove_stopwords:
						seg = [word for word in seg if word not in self.stopwords]
					self.seg_data.append(seg)

			logging.info("%d %ss have been read in." % (len(self.seg_data), seg_type))




