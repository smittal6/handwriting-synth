import numpy as np
import torch
import torch.nn as nn
from models.model import UnconditionedHand
from models.con_model import ConditionedHand
from utils.dataloader import StrokesDataset

strokes = np.load('../data/strokes.npy')
stroke = strokes[0]


def get_testinput(dataset,text):
    """
    Get the input, and encoding for the text.
    """
    start_stroke = np.asarray((0,0,0))
    start = torch.from_numpy(start_stroke).float()
    start = Variable(start)
    start = start.view(-1,1,3)
    encoding = dataset.getOneHot(text)
    encoding = Variable(torch.from_numpy(encoding))
    encoding = encoding.float()
    return start,encoding



def generate_unconditionally(random_seed=1):
    # Input:
    #   random_seed - integer
    # Output:
    #   stroke - numpy 2D-array (T x 3)
    dataset = StrokesDataset()
    mod = UnconditionedHand()
    mod.load_state_dict(torch.load('./save/uncon.model'))
    input,encoding = get_testinput(dataset,text='dummy')
    stroke = mod.get_stroke(input)
    stroke = stroke.astype(np.float32)
    return stroke


def generate_conditionally(text='welcome to lyrebird', random_seed=1):
    # Input:
    #   text - str
    #   random_seed - integer

    # Output:
    #   stroke - numpy 2D-array (T x 3)
    dataset = StrokesDataset()
    mod = ConditionedHand(dataset.vec_len)
    mod.load_state_dict(torch.load('./save/conditioned.model'))
    test_in,encoding = get_testinput(dataset,text)
    stroke = mod.get_stroke(test_in,encoding)
    return stroke


def recognize_stroke(stroke):
    # Input:
    #   stroke - numpy 2D-array (T x 3)

    # Output:
    #   text - str
    return 'welcome to lyrebird'
