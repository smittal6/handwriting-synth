import numpy as np
import torch
import torch.nn as nn
from models.model import UnconditionedHand
from models.con_model import ConditionedHand
from utils.dataloader import StrokesDataset

strokes = np.load('../data/strokes.npy')
stroke = strokes[0]



def get_testinput():
    start_stroke = np.asarray((0,0,0))
    start = torch.from_numpy(start_stroke).float()
    start = Variable(start)
    start = start.view(-1,1,3)
    return start



def generate_unconditionally(random_seed=1):
    # Input:
    #   random_seed - integer
    # Output:
    #   stroke - numpy 2D-array (T x 3)
    dataset = StrokesDataset()
    mod = UnconditionedHand()
    mod.load_state_dict(torch.load('./save/uncon.model'))
    input = get_testinput()
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
    mod = ConditionedHand()
    return stroke


def recognize_stroke(stroke):
    # Input:
    #   stroke - numpy 2D-array (T x 3)

    # Output:
    #   text - str
    return 'welcome to lyrebird'
