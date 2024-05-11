import logging
import os
import linecache
from .evaluator import Evaluator
from .bm25 import BM25Matcher


def getMatcher(input_path, matcher_type, rm_sw):
    output_path = ""
    if matcher_type == "bm25":
        cur_dir = os.getcwd()
        os.chdir(os.path.dirname(__file__))
        if input_path.endswith(".txt"):
            output_path = input_path[:-4] + '_seg.txt'
        matcher = BM25Matcher(input_path=input_path, output_path=output_path, remove_stopwords=rm_sw)
        os.chdir(cur_dir)
        return matcher
    else:
        raise Exception("Invalid matcher type: {}.".format(matcher_type))


class Responder:
    def __init__(self, input_path, matcher_type="bm25", rm_sw=False):
        self.general_questions = []
        self.path = os.path.dirname(__file__)
        self.matcher = getMatcher(input_path, matcher_type, rm_sw)
        self.ans_path = os.path.join(self.path, "data/answer.txt")

        topic_path = os.path.join(self.path, "topic/output/model.h5")
        sbert_path = os.path.join(self.path, 'distiluse-base-multilingual-cased-v1')
        self.evaluator = Evaluator(topic_path=topic_path, sbert_path=sbert_path)

        self.moduleTest()

    def getResponse(self, query, debug=False):
        indexes, _ = self.matcher.match(query)
        title_list = []
        ans_list = []

        for index in indexes:
            title = ''.join(self.matcher.seg_data[index])
            ans = linecache.getline(self.ans_path, index + 1)
            title_list.append(title)
            ans_list.append(ans)

        return self.evaluator.getBestResponse(query, title_list, ans_list, debug)

    def getBm25Response(self, query):
        indexes, scores = self.matcher.match(query)
        if len(indexes) == 0:
            return None, 0

        ans = linecache.getline(self.ans_path, indexes[0] + 1)
        return ans, scores[0]

    def moduleTest(self):
        logging.info("Testing modules...")

        try:
            self.matcher.segment_word("Testing segmentation module", False)
            logging.info("Testing successfully！")
        except Exception as ex:
            logging.info(repr(ex))
            logging.info("Modules loading failed.")
