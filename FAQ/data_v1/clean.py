import re
import os
import time
import json
import heapq
from util import *


def filter_break(text):
    return text.replace('\n', '').replace('\r', '').replace(' ', '')


def filter_str(desstr,restr=''):
    return res.sub(restr, desstr)


def clean_answer(text, max_len=200, min_len=5):
    for f in filters:
        if f.search(text):
            return None
    text = filter_break(text)
    text = filter_str(text)
    for p in patterns:
        text = p.sub("", text)

    if len(text) < min_len:
        return None

    if len(text) > max_len:
        sentences = split_sentence.split(text)
        best_len = min_len
        text = None
        for s in sentences:
            size = len(s)
            if size < max_len and size > best_len:
                best_len = size
                text = s
    return text


def get_label(labels):
    label_dict= {'成长': '成长', '自我成长': '成长', '成长过程': '成长', '学生成长': '成长', '自我接纳': '成长', '人格特质': '成长', '性格完善': '成长', '心理危机': '成长', '心理咨询': '成长', '发展规律': '成长', 
    '治疗': '治疗', '疾病诊断': '治疗', '病态人格': '治疗', '治疗方法': '治疗', '躯体反应': '治疗', '疗愈方法': '治疗', '精神障碍': '治疗', '创伤治疗': '治疗', 
    '恋爱': '恋爱', '恋爱经营': '恋爱', '单身': '恋爱', '出轨': '恋爱', '失恋': '恋爱', '好感': '恋爱', '挽回前任': '恋爱', '性取向': '恋爱', '性心理': '恋爱', '性生活': '恋爱', 
    '情绪': '情绪', '情绪调节': '情绪', '抑郁情绪': '情绪', '焦虑': '情绪', '焦虑情绪': '情绪', '压力管理': '情绪', '恐慌无助': '情绪', '表达情绪': '情绪', '情绪智力': '情绪', '困惑': '情绪', '安全感': '情绪', '脆弱流泪': '情绪', '内疚羞耻': '情绪', 
    '婚姻': '婚姻', '婚姻经营': '婚姻', '婚姻观念': '婚姻', '离婚': '婚姻', '婚前': '婚姻', '产前产后': '婚姻', 
    '人际': '人际', '人际边界': '人际', '朋友': '人际', '社交恐惧': '人际', '舍友同学': '人际', '交往模式': '人际', '欺骗与信任': '人际', '社会适应': '人际', '吵架': '人际', '沟通': '人际', '矛盾冲突': '人际', 
    '行为': '行为', '熬夜': '行为', ' 逃避': '行为', '拖延': '行为', 
    '职业': '工作学习', '工作学习': '工作学习', '工作压力': '工作学习', '职场人际': '工作学习', '职业管理': '工作学习', '择业技巧': '工作学习', ' 中年危机': '工作学习', '考证读研': '工作学习', 
    '家庭': '家庭', '家庭创伤': '家庭', '家庭关系': '家庭', '父母沟通': '家庭', '子女沟通': '家庭', '婆媳岳婿': '家庭', '家人控制': '家庭', 
    '解梦': '梦', '梦的解析': '梦', 
    '科普': '讨论', '热点': '讨论', ' 热点话题': '讨论', '书籍': '讨论', '公共事件': '讨论'}
    new_labels = set()
    for i in labels:
        if i in label_dict.keys():
            new_label = label_dict[i]
            new_labels.add(new_label)
    return new_labels


def get_answer(raw, only=True):
    # first choice: recommended answer
    ans_list = [x for x in raw if x['recommend_flag'] =='推荐']
    results = []
    # only one response is selected
    if only is True and len(ans_list) > 1:
        ans_list = heapq.nlargest(1, ans_list, key=lambda item: int(item['zan']))

    # second choice: favorate answer
    if len(ans_list) == 0:
        # ans_list = heapq.nlargest(1, ans_list, key=lambda item: int(item['zan']))
        most_like = 0
        favorate_idx = -1
        for idx, item in enumerate(raw):
            if int(item['zan']) > most_like:
                favorate_idx = idx
                most_like = int(item['zan'])
        if favorate_idx != -1:
            ans_list.append(raw[favorate_idx])

    for item in ans_list:
        ans = item['content'].replace("|", '"')
        ans = clean_answer(ans)
        if ans is not None:
            results.append(ans)

    return results


def get_question(title, content, max_len=200, min_len=5):
    if title in content:
        q = content
    elif content in title:
        q = title
    else:
        q = title + " " + content
    
    for f in filters:
        if f.search(q):
            return None
    
    q =  re.sub("(如题|RT|rt|求[助解]|求指导|求请教)[{}]?".format(punc), "", q)
    q = filter_break(q)

    if len(q) < min_len or len(q) > max_len:
        return None
    return q


if __name__ == '__main__':
    start_time = time.time()
    data_path = os.path.join(os.getcwd(), 'data')
    title_file = open(os.path.join(data_path, 'title.txt'), 'r', encoding='utf-8')
    content_file = open(os.path.join(data_path, 'content.txt'), 'r', encoding='utf-8')
    answer_file = open(os.path.join(data_path, 'answer.txt'), 'r', encoding='utf-8')
    label_file = open(os.path.join(data_path, 'label.txt'), 'r', encoding='utf-8')
    question_file = open(os.path.join(data_path, 'question.txt'), 'w', encoding='utf-8')
    best_answer_file = open(os.path.join(data_path, 'best_answer.txt'), 'w', encoding='utf-8')
    new_label_file = open(os.path.join(data_path, 'new_label.txt'), 'w', encoding='utf-8')

    count = 0
    for title, content, answer, label in zip(title_file, content_file, answer_file, label_file):
        q = get_question(title.strip(), content.strip())
        answer = answer.replace('"', "|")
        answer = answer.replace("'", '"')
        try:
            answers = json.loads(answer.strip("\n"))
        except ValueError:
            continue

        ans = get_answer(answers)
        labels= label.strip().split(",")
        labels = get_label(labels)

        if q is None or len(ans) == 0 or len(labels) == 0:
            continue

        for i in ans:
            question_file.write(q + '\n')
            best_answer_file.write(i + '\n')
            new_label_file.write(','.join(labels) + '\n')
            count += 1
            if count % 5000 == 0:
                print("Process: %d" % count)
        
    title_file.close()
    content_file.close()
    question_file.close()
    answer_file.close()
    best_answer_file.close()

    end_time = time.time()
    print("Total Process: %d, Consumed Time: %.2fs." %(count, end_time - start_time))

