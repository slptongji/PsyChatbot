import os
import FAQ.qa as qa
import sys


class Chatbot:
    def __init__(self, input_path, name="PsyChatbot", matcher_type="bm25", rm_sw=False):
        self.name = name
        self.responder = qa.Responder(input_path, matcher_type, rm_sw)

    def getResponse(self, query, threshold=30, debug=False):
        res, sim = self.responder.getResponse(query, debug)
        if sim > threshold:
            return res, sim
        else:
            return None, 0

    def getBm25Response(self, query):
        return self.responder.getBm25Response(query)


if __name__ == "__main__":
    cur_path = os.getcwd()
    sys.path.append(cur_path)
    input_path = "data/question.txt"
    input_path2 = "data/answer.txt"
    chatbot = Chatbot(input_path=input_path, rm_sw=True)

    print("Initializing conversation.")

    while True:
        raw = input('>>')
        if raw == 'quit':
            exit()
        reply, sim = chatbot.getResponse(raw, debug=True)
        print("Response: %s" % reply)
        print("Similarity: %d" % sim)
        # print("-------------------------")
        # reply, sim = chatbot.getBm25Response(raw)
        # print("Response 2: %s" % reply)
        # print("Similarity 2: %d" % sim)

