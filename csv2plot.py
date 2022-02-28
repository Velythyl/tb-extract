import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt

from utils import run


def do_csv(path, num_runs):
    frame = pd.read_csv(path)

    name = frame['name'][1]
    #value = frame['value']

    x_values = frame['step'].unique()
    smoothings = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

    for smoothing in smoothings:
        acc = []
        for run_id in range(num_runs):
            run = frame.iloc[run_id::3,:]
            if smoothing == 0:
                acc.append(run['value'])
            else:
                acc.append(run.ewm(alpha=(1-smoothing)).mean()['value'])

        acc = np.array(acc)
        y_values = acc.mean(axis=0)
        y_errors = acc.std(axis=0)


        ax = sns.lineplot(x_values, y_values)
        ax.fill_between(x_values, y_values-y_errors, y_values+y_errors, alpha=0.5)
        plt.grid(alpha=0.3)
        plt.show()
    #plt.show()

    #grouped = frame.groupby('step')
    #means = grouped.mean()
    #stds = grouped.std()


"""
on the same plot:
    adv and coop
    
all envs

dash horizontal for newer paper's and paired's top performance (from the table)
"""

    #
    #def plots(smoothing):
    #    #plt.plot(values["value"], alpha=0.4)


    #plots(0.5)

if __name__ == "__main__":
    path = "/home/velythyl/Desktop/tb-extract"
    all_paths = run(f"find {path} -name 'out.csv'")

    do_csv("/home/velythyl/Desktop/tb-extract/out/none_none_MinigridSettable-v0_cooperative_maxnature_maxlearner_randompicks[agent]/Eval_MultiGrid-SixteenRooms-v0/solved_rate/out.csv", 3)