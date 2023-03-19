import matplotlib.pyplot as plt
import pandas as pd



def csv_painter(path):
    print("begin to paint table for csv file.")
    data = pd.read_csv(path)
    # all is 1-index
    epoch_num = max(data['epoch'])
    step_num = max(data['step'])
    agent_set = len(set(data['agent']))
    for epoch in epoch_num:
        



def _accurancy_distance():
    """
    handle the accurany of landmark points distance
    """
    plt.xlabel("epoch")
    plt.ylabel("")


def _accurancy_occupation():
    """success_num/epoch_num
    """


