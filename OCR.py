import argparse
import json
from typing import Tuple, List
import os
import cv2
import editdistance
from path import Path
from src.main import infer
from src.main import getModel
from src.model import Model, DecoderType




class FilePaths:
    """Filenames and paths to data."""
    fn_char_list = 'model/charList.txt'
    fn_summary = 'model/summary.json'
    fn_corpus = 'data/corpus.txt'





img_file = "data/word.png"
model = getModel()
infer(model, img_file)
