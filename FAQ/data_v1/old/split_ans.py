import os
import math
import time

start = time.time()

count = 0
data_path = os.getcwd()
process_data_path = os.path.join(data_path, 'ans')
answer_file = os.path.join(data_path, 'raw_ans200.txt')

with open(answer_file, 'r', encoding='utf-8') as f:
	lines = f.readlines()
	count = len(lines)
	for index, line in enumerate(lines, 1):
		wf = os.path.join(process_data_path, "%d.txt" % math.ceil(index / 1000))
		with open(wf, 'a', encoding='utf-8') as tmp:
			tmp.write(line)
		if index % 10000 == 0:
			print("已完成%d条数据的处理" % index)

end = time.time()
print("总共完成%d条数据的处理，耗时 %.2f s" % (count,(end-start)))