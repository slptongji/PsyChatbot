import os
import sys
import torch
import numpy as np
from keras.models import load_model
# from keras import backend as K
from sentence_transformers import SentenceTransformer, util
sys.path.append(os.path.join(os.path.dirname(__file__), "topic"))
from .matcher import Matcher
from .topic.train import Attention
from .topic.albert_zh.extract_feature import BertVector as AlbertVector

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["KERAS_BACKEND"] = "tensorflow"


class Evaluator(Matcher):
    def __init__(self, topic_path, sbert_path, max_seq_len=200):
        super().__init__()
        # K.clear_session()
        self.path = os.path.dirname(__file__)
        self.topic_model = load_model(topic_path, custom_objects={"Attention": Attention})
        self.albert = AlbertVector(pooling_strategy="NONE", max_seq_len=max_seq_len)
        self.sentence_bert = SentenceTransformer(sbert_path)

        # sbert_path = os.path.join(self.path, "data/encode_title.pt")
        # if not os.path.exists(sbert_path):
        #     li = []
        #     count = 0
        #     with open('data/seg_title200.txt', 'r', encoding='utf-8') as f:
        #         for line in f:
        #             line = line.strip()
        #             embedding = self.sentence_bert.encode(line, convert_to_tensor=True)
        #             li.append(embedding)
        #             count += 1
        #             if count % 1000 == 0:
        #                 print("Processed: %d." % count)
        #     print("Total processed: %d." % count)                   
        #     torch.save(li, sbert_path)

        # self.encode_titles = torch.load(os.path.join(self.path, "data/encode_title.pt"), map_location='cpu')

    def getBestResponse(self, query, titles, ans, debug):
        if len(titles) == 0:
            return None, 0

        # topic similarity
        q_topic = self.getTopicVector(query)

        t_topic = []
        for title in titles:
            topic = self.getTopicVector(title)
            t_topic.append(topic)
        t_topic = np.array(t_topic).reshape(len(titles), -1)
        topic_score = np.sum(q_topic * t_topic, axis=1) / (np.linalg.norm(t_topic, axis=1) * np.linalg.norm(q_topic))

        # question similarity
        embedding1 = self.sentence_bert.encode(query, convert_to_tensor=True)
        embeddings2 = self.sentence_bert.encode(titles, convert_to_tensor=True)
        sentence_score = util.pytorch_cos_sim(embedding1, embeddings2)
        sentence_score = np.array(sentence_score).flatten()

        # final score
        scores = 30 * topic_score + 70 * sentence_score
        topk_idx = np.argsort(scores)[::-1]
        best_idx = topk_idx[0]

        if debug:
            print("Best match question: {}".format(titles[best_idx]))
            print('Origin candidate questions:')
            for i, title in enumerate(titles[:3]):
                print('Candidate {}: {}'.format(i+1, title))

        return ans[best_idx], scores[best_idx]

    def getTopicVector(self, sentence):
        vec = self.albert.encode([sentence])["encodes"][0]
        vec = np.array([vec])
        pred = self.topic_model.predict(vec)[0]
        return pred

    def getBestResponseTest(self, query, titles):
        if len(titles) == 0:
            return None, 0

        # topic similarity
        q_topic = self.getTopicVector(query)

        t_topic = []
        for title in titles:
            topic = self.getTopicVector(title)
            t_topic.append(topic)
        t_topic = np.array(t_topic).reshape(len(titles), -1)
        topic_score = np.sum(q_topic * t_topic, axis=1) / (np.linalg.norm(t_topic, axis=1) * np.linalg.norm(q_topic))

        # question similarity
        embedding1 = self.sentence_bert.encode(query, convert_to_tensor=True)
        embeddings2 = self.sentence_bert.encode(titles, convert_to_tensor=True)
        sentence_score = util.pytorch_cos_sim(embedding1, embeddings2)
        sentence_score = np.array(sentence_score).flatten()

        # final score
        scores = 30 * topic_score + 70 * sentence_score
        topk_idx = np.argsort(scores)[::-1]

        if topk_idx[0] != 0:
            print('{}: {}, {}:{}'.format(topk_idx[0], scores[topk_idx[0]], topk_idx[1], scores[topk_idx[1]]))

        return topk_idx