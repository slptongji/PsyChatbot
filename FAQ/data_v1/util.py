import re


punc = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~“”？，！【】（）、。：；’‘……￥·～"""
res = re.compile(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a\’!\"#$%&\'()*+,-./:;<=>?@，。?、…【】《》？“”‘’！["u"\\]^_`{|}~\s])")

filters = [
    re.compile("(电话|微信|QQ).?\d"),
    re.compile("昵称是本人的工作平台qq"),
    re.compile("壹心理")
]


patterns = [
    re.compile("[楼题]主(你)?好(呀)?[{}]".format(punc)),
    re.compile("给[楼题]主温暖的抱抱[{}]".format(punc)),
    re.compile("(心疼|抱抱)(楼主|题主|你)[{}]".format(punc)),
    re.compile("[感谢]谢(你的)?邀请[{}]".format(punc)),
    re.compile("来听.*?说说[{}]".format(punc)),
    re.compile("(你好)?我是(心理)?咨询师.+?[{}]".format(punc)),
    re.compile("我是鲸鱼社工.+?[{}]".format(punc)),
    re.compile("[{}]?壹心理鲸鱼社工[{}]?".format(punc, punc)),
    re.compile("关注情感导师.*"),
    re.compile("(回复)?@.*?([{}]|\s)".format(punc)),
    re.compile("你好[{}]".format(punc)),
]



split_sentence = re.compile(r'。|！|？|～|\.|!|\?|~')
