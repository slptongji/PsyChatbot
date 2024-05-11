import json
import numpy as np
from keras.models import load_model

from att import Attention
from albert_zh.extract_feature import BertVector
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="1"

if __name__ == "__main__":
    topic_model = load_model("model.h5", custom_objects={"Attention": Attention})
    bert_model = BertVector(pooling_strategy="NONE", max_seq_len=200)
    print("Start prediction.")

    while True:
        raw = input('>>')
        if(raw == 'quit'):
            exit()
        text =  raw.replace("\n", "").replace("\r", "").replace("\t", "")
        vec = np.array([bert
        _model.encode([text])["encodes"][0]])
        predicted = topic_model.predict(vec)[0]
        pred_max = np.max(predicted)
        pred_min = np.min(predicted)
        predicted = (predicted - pred_min)/(pred_max - pred_min)
        indices = [i for i in range(len(predicted)) if predicted[i] > 0.5]

        with open("label_type.json", "r", encoding="utf-8") as f:
            labels = json.loads(f.read())
            print("Sentence: %s" % text)
            print("Topic": %s" % ",".join([labels[index] for index in indices]))
