import jieba
import logging
import aiml
from demo import Chatbot
import re

jieba.setLogLevel(logging.INFO)


def filter_space(text):
    space_pattern = re.compile(r'([\u4e00-\u9fa5])\s + ([\u4e00-\u9fa5])')
    tmp = space_pattern.sub(r'\1\2', text)
    return space_pattern.sub(r'\1\2', tmp)


class LayerFilter:
    def __init__(self, input_path, rm_sw=False):
        self.default = ["原来如此。", "谢谢你告诉我这么多。", "摸摸。", "噢，别担心，这很正常。", "我明白了。那你觉得这说明了什么？",
                        "这让你感觉如何？", "其他人怎么说？", "还有吗？", "然后呢？", "嗯。", "确实。"]
        self.pipeline = [Template(), Retrieval(input_path, rm_sw=False)]
        self.context = None

    def get_response(self, query):
        temp_res = self.pipeline[0].search_response(query)

        if "[DEFAULT]" not in temp_res:
            return temp_res
        else:
            default = temp_res.lstrip('[DEFAULT]')

        retr_res = self.pipeline[1].search_response(query)
        if retr_res:
            return retr_res

        return default


class BaseLayer:
    def __init__(self):
        self.logger = logging.getLogger()

    def print_log(self, msg):
        self.logger.info(msg)


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
    def __init__(self, input_path, rm_sw=False):
        super(Retrieval, self).__init__()
        self.chatbot = Chatbot(input_path, rm_sw)
        self.print_log('Retrieval Layer is ready.')

    def search_response(self, query):
        reply, sim = self.chatbot.getResponse(query)
        return reply
