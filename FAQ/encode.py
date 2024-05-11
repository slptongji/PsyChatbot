import numpy as np
import torch
from sentence_transformers import SentenceTransformer

if __name__ == '__main__':
    sentence_bert = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    output_path = 'data/encode_ans.pt'

    with open('data/seg_ans200.txt', 'r', encoding='utf-8') as f:
        li = []
        count = 0
        for line in f:
            line = line.strip()
            embedding = sentence_bert.encode(line, convert_to_tensor=True)
            li.append(embedding)
            count += 1
            if count % 1000 == 0:
                print("Processed: %d." % count)
        print("Total processed: %d." % count)
            
    torch.save(li, output_path)
