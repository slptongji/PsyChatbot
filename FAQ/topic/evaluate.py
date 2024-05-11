import json
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.metrics import hamming_loss, classification_report, jaccard_score, accuracy_score, f1_score
from train import Attention
from albert_zh.extract_feature import BertVector


def print_evaluate_score(y_val, predicted):
    accuracy = accuracy_score(y_val, predicted)
    f1_score_macro = f1_score(y_val, predicted, average='macro')
    f1_score_micro = f1_score(y_val, predicted, average='micro')
    f1_score_weighted = f1_score(y_val, predicted, average='weighted')
    print("Accuracy: {}".format(accuracy))
    print("F1_score_macro: {}".format(f1_score_macro))
    print("F1_score_micro: {}".format(f1_score_micro))
    print("F1_score_weighted: {}".format(f1_score_weighted))


def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        tmp = None

        if len(set_true) == 0 and len(set_pred) == 0:
            tmp = 1
        else:
            tmp = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        acc_list.append(tmp)
    return np.mean(acc_list)


def predict_single_text(text):
    vec = np.array([bert_model.encode([text])["encodes"][0]])
    pred_y = model.predict(vec)[0]

    pred_max = np.max(pred_y)
    pred_min = np.min(pred_y)
    pred_y = (pred_y - pred_min) / (pred_max - pred_min)

    pred_y = pred_y.tolist()
    one_hot = [0] * len(event_type_list)
    if max(pred_y) <= 0.5:
        indexes = [len(pred_y)]
    else:
        pred_y[-1] = 0 
        indexes = [i for i in range(len(pred_y)) if pred_y[i] > 0.5]
    for idx in indexes:
        one_hot[idx] = 1
    # indices = [i for i in range(len(pred_y)) if pred_y[i] > 0.5]
    # one_hot = [0] * len(event_type_list)
    # for index in indices:
    #     one_hot[index] = 1

    return pred_y, one_hot, ",".join([event_type_list[index] for index in indexes])


def evaluate():
    with open("./data/test.txt", "r", encoding="utf-8") as f:
        content = [_.strip() for _ in f.readlines()]

    pred_rank_list = []
    pred_list = []
    true_y_list, pred_y_list = [], []
    true_label_list, pred_label_list = [], []

    for i in range(len(content)):
        if i + 1 % 5000 == 0:
            print("Predicted: {}".format(i+1))
        true_label, text = content[i].split(" ", maxsplit=1)
        true_y = [0] * len(event_type_list)
        for i, event_type in enumerate(event_type_list):
            if event_type in true_label:
                true_y[i] = 1

        pred, pred_y, pred_label = predict_single_text(text)

        pred_rank = np.argsort(pred)[::-1]

        true_y_list.append(true_y)
        pred_y_list.append(pred_y)
        true_label_list.append(true_label)
        pred_label_list.append(pred_label)

        pred_rank_list.append(pred_rank)
        pred_list.append(pred)

    print("Total predict: {}".format(len(content)+1))

    pred_rank_list = np.array(pred_rank_list)
    np.save("./output/pred_rank.npy", pred_rank_list)
    pred_list = np.array(pred_list)
    np.save("./output/pred.npy", pred_list)

    print(classification_report(true_y_list, pred_y_list, digits=4))
    print_evaluate_score(true_y_list, pred_y_list)
    return true_label_list, pred_label_list, hamming_loss(true_y_list, pred_y_list), jaccard_score(true_y_list, pred_y_list, average ='samples')


if __name__ == "__main__":
    model = load_model("./output/model.h5", custom_objects={"Attention": Attention})
    with open("./data/topic_type.json", "r", encoding="utf-8") as f:
        event_type_list = json.loads(f.read())
    bert_model = BertVector(pooling_strategy="NONE", max_seq_len=200)
    true_labels, pred_lables, h_loss, samples_jaccard = evaluate()
    df = pd.DataFrame({"y_true": true_labels, "y_pred": pred_lables})
    df.to_csv("./output/pred_result.csv")

    print("Jaccard score: %s" % samples_jaccard)
    print("Hamming loss: %s" % h_loss)
