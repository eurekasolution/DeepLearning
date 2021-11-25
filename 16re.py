#16re.py (정규식을 이용한 데이터 튜닝 후 학습)

from tensorflow.keras.preprocessing.text import Tokenizer
from konlpy.tag import Okt

import re
from tqdm import tqdm
from pprint import pprint
import nltk

import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import numpy as np
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# debug, warning, info, error, danger, critical (7)