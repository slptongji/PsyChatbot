import random
import jieba.analyse as analyse
import logging
import aiml
import re
import FAQ.qa as qa
from collections import deque


def filter_space(text):
    space_pattern = re.compile(r'([\u4e00-\u9fa5])\s + ([\u4e00-\u9fa5])')
    tmp = space_pattern.sub(r'\1\2', text)
    return space_pattern.sub(r'\1\2', tmp)


class PsyChatbot:
    def __init__(self, input_path, rm_sw=False, context_len=10):
        self.countdown = 0
        self.default = ["原来如此。", "谢谢你告诉我这么多。", "摸摸。", "噢，别担心，这很正常。", "我明白了。那你觉得这说明了什么？",
                        "这让你感觉如何？", "其他人怎么说？", "还有吗？", "然后呢？", "嗯。", "确实。"]
        self.pipeline = [Template(), Retrieval(input_path, rm_sw=rm_sw)]
        self.context = deque(maxlen=context_len)

    def run(self):
        print("PsyChatbot is ready. Let's start.")
        print('')
        print('----------------------------------')
        print('')

        response = self.get_response("你好")
        print("PsyChatbot: %s" % response)

        while True:
            message = input('User>>')
            if message == 'quit' or message == "再见":
                response = self.get_response('再见')
                print("PsyChatbot: %s" % response)
                exit()

            self.context.append(message)
            self.countdown += 1
            response = self.get_response(message).strip()
            print("PsyChatbot: %s" % response)

            if self.countdown >= random.randint(5, 8):
                self.countdown = 0
                print("PsyChatbot: %s（好/不好）" % self.get_response("STATE"))

    def get_response(self, query):
        temp_res = self.pipeline[0].search_response(query)

        if "[DEFAULT]" not in temp_res:
            return temp_res
        else:
            default = temp_res.lstrip('[DEFAULT]')

        # contexts = " ".join(self.context)
        # keywords = analyse.extract_tags(contexts, topK=5)
        # query = " ".join(keywords) + " " + query

        retr_res = self.pipeline[1].search_response(query)
        if retr_res:
            return retr_res

        return default


class BaseLayer:
    def __init__(self, log=True):
        self.logger = logging.getLogger()
        if not log:
            self.close_log()

    def print_log(self, msg):
        self.logger.info(msg)

    def search_response(self, query):
        ...


class Template(BaseLayer):
    def __init__(self):
        super(Template, self).__init__()
        self.chatbot = aiml.Kernel()
        self.chatbot.learn("startup.xml")
        self.chatbot.respond("LOAD AIML")
        self.print_log('Template Layer is ready.')

    def search_response(self, query):
        res = self.chatbot.respond(query)
        return filter_space(res)


class Retrieval(BaseLayer):
    def __init__(self, input_path, matcher_type="bm25", rm_sw=False):
        super(Retrieval, self).__init__()
        self.responder = qa.Responder(input_path, matcher_type, rm_sw)
        self.print_log('Retrieval Layer is ready.')

    def search_response(self, query):
        reply, sim = self.getResponse(query)
        return reply

    def getResponse(self, query, threshold=30):
        res, sim = self.responder.getResponse(query)
        if sim > threshold:
            return res, sim
        else:
            return None, 0

    def getBm25Response(self, query):
        return self.responder.getBm25Response(query)


if __name__ == "__main__":
    input_path = "data/question.txt"
    mybot = PsyChatbot(input_path, rm_sw=True)
    mybot.run()
