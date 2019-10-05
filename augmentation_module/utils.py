# -*- coding: utf-8 -*-

import mojimoji
import re


digit_pattern = re.compile(r'[1-9]+')


def normalize(sentence):
    normalized = mojimoji.zen_to_han(sentence, kana=False)
    return digit_pattern.sub('#', normalized)
