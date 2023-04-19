import os
import json
import time

def filter_break(text):
    return text.replace('\n', ' ').replace('\r', ' ')


if __name__ == '__main__':
    start_time = time.time()
    json_data = open("ques_ans1.json", encoding="utf-8").read()
    data = json.loads(json_data)

    count = 0
    data_path = os.path.join(os.getcwd(), 'data')
    title_file = open(os.path.join(data_path, 'title.txt'), 'w', encoding='utf-8')
    content_file = open(os.path.join(data_path, 'content.txt'), 'w', encoding='utf-8')
    answer_file = open(os.path.join(data_path, 'answer.txt'), 'w', encoding='utf-8')
    label_file = open(os.path.join(data_path, 'label.txt'), 'w', encoding='utf-8')

    for item in data:
        info = item['ques_info']
        # filter questions without answer
        if int(info['answer_count']) <= 0:
            continue

        # get useful information
        title = filter_break(info['title'])
        content = filter_break(info['content'])
        label = ','.join(info['ques_label'])
        answer = filter_break(str(item['answers_info']))

        # write in
        title_file.write(title+"\n")
        content_file.write(content+"\n")
        answer_file.write(answer+"\n")
        label_file.write(label+"\n")

        count += 1
        if count % 10000 == 0:
            print("Processed: %d" % count)

    end_time = time.time()
    print("Total Process: %d, Consumed Time: %.2f s" % (count,(end_time-start_time)))
    # close files
    title_file.close()
    content_file.close()
    answer_file.close()
    label_file.close()