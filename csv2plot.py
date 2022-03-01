import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt

from utils import run

smoothings = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
DEBUG = False

def do_csv(path, csv_paths, num_runs):
    csv_dict = {}

    for csv in csv_paths:
        try:
            csv_path = f"{path}/{csv}"

            frame = pd.read_csv(csv_path)

            name = frame['name'][1]
            #value = frame['value']

            x_values = frame['step'].unique()

            x_y_e = []
            for smoothing in smoothings:
                acc = []
                for run_id in range(num_runs):
                    run = frame.iloc[run_id::3,:]
                    if smoothing == 0:
                        acc.append(run['value'])
                    else:
                        acc.append(run.ewm(alpha=(1-smoothing)).mean()['value'])

                acc = [ac.to_numpy() for ac in acc]
                acc = np.array(acc).astype(float)
                y_values = acc.mean(axis=0)
                y_errors = acc.std(axis=0)

                if DEBUG:
                    ax = sns.lineplot(x_values, y_values)

                    x_labels = x_values[::2]
                    plt.locator_params(axis='y', nbins=10)
                    #ax.set_yticks()
                    ax.set_xticks(x_labels, x_labels, ha='right', rotation=45, rotation_mode="anchor")
                    #ax.set_xticklabels(list(map(str, x_values)))
                    ax.fill_between(x_values, y_values - y_errors, y_values + y_errors, alpha=0.5)
                    plt.xlabel("Timestep")

                    plot_name, y_axis_name = name.replace("Eval_", "").replace("MultiGrid-", "").replace("-Minigrid", "").replace("-v0", "").split("/")


                    plt.title(plot_name)
                    plt.ylabel(y_axis_name)
                    plt.tight_layout()
                    plt.grid(alpha=0.3)
                    plt.show()
                    exit()

                x_y_e.append((x_values, y_values, y_errors, name))
            csv_dict[csv] = x_y_e

        except ValueError as e:
            if 'setting an array element with a sequence' in e.args[0]:
                assert "teacher" or "last_step" in csv.lower()
            else:
                print(f"WARNING: assumed CSV {csv} was invalid and skipped it.")
                raise e

    return csv_dict

def just_one_plot(x_values, y_values, y_errors, name):


def plot(data, keys, names):
    for key in keys:

        if "Eval" and "solve" not in key:
            continue

        for smoothing_id in range(len(smoothings)):
            fig = plt.figure(figsize=(6,6))
            for dict_id, dict in enumerate(data):
                packing = dict[key]
                x_values, y_values, y_errors, name = packing[smoothing_id]

                line_label = ""
                if "cooperative" in names[dict_id]:
                    line_label = "cooperative"
                if "adversarial" in names[dict_id]:
                    line_label = "adversarial"

                ax = sns.lineplot(x_values, y_values, label=line_label)

                x_ticks = x_values[::2]
                x_labels = map(lambda x: f"{str(1000000/x)[:3]}M", x_ticks)
                plt.locator_params(axis='y', nbins=10)

                y_ticks = np.arange(10)/10
                ax.set_yticks(y_ticks, y_ticks)
                ax.set_xticks(x_ticks, x_ticks, ha='right', rotation=45, rotation_mode="anchor")
                # ax.set_xticklabels(list(map(str, x_values)))
                ax.fill_between(x_values, y_values - y_errors, y_values + y_errors, alpha=0.5)
            plt.xlabel("Timestep")

            plot_name, y_axis_name = name.replace("Eval_", "").replace("MultiGrid-", "").replace("-Minigrid",
                                                                                                 "").replace("-v0",
                                                                                                             "").split(
                "/")

            plt.title(plot_name)

            y_axis_name = {
                "solved_rate": "Solve Rate",
                "median": "Median"
            }[y_axis_name]

            plt.ylabel(y_axis_name)
            plt.tight_layout()
            plt.grid(alpha=0.3)

            plt.savefig(f"{names[0]}/{plot_name}_{smoothings[smoothing_id]}.png", dpi=fig.dpi)
            #plt.show()
            #exit()

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

def find_all_experiment_pairs(path):
    all_paths = run(f"find {path} -name 'out.csv'").split("\n")

    experiment_names = set(map(lambda x: x.replace(path, "").split("/")[1], all_paths))

    path_matchings = dict()
    for sub_paths in experiment_names:
        pattern = sub_paths.replace("cooperative", "PATTERN").replace("adversarial", "PATTERN")
        if pattern not in path_matchings:
            path_matchings[pattern] = []
        path_matchings[pattern].append(sub_paths)

    for key, values in path_matchings.items():
        assert len(values) == 2 or "blocks,goal,agent" in values[0]

    experiment_names = []
    for pair in path_matchings.values():
        experiment_names.append(list(map(lambda x: f"{path}/{x}", pair)))

    return experiment_names

def find_all_experiment_paths(path):
    all_paths = run(f"find {path} -name 'out.csv'").split("\n")
    csv_paths = set(map(lambda x: "/".join(x.replace(path, "").split("/")[2:]), all_paths))
    return csv_paths

def do_mp(pair, csv_paths):
    print("PAIR")
    print(pair)

    result = [do_csv(pair[0], csv_paths, 3)]
    if len(pair) == 2:
        result.append(do_csv(pair[1], csv_paths, 3))
        keys2 = set(result[1].keys())
    else:
        keys2 = set()

    keys1 = set(result[0].keys())
    uh = keys1 == keys2  # if we dont do this, sets are counted as empty... fuck python

    common_keys = keys1.intersection(keys2)

    plot(result, common_keys, pair)


if __name__ == "__main__":
    path = "/home/velythyl/Desktop/tb-extract/out"

    sns.set_palette("colorblind")
    matplotlib.rcParams.update({'font.size': 14})
    csv_paths = find_all_experiment_paths(path)
    exp_pairs = find_all_experiment_pairs(path)


    import multiprocessing as mp
    pool = mp.Pool()
    result = pool.starmap(do_mp, zip(exp_pairs, [csv_paths] * len(exp_pairs)))
    pool.close()
    pool.join()


   # do_csv("/home/velythyl/Desktop/tb-extract/out/none_none_MinigridSettable-v0_cooperative_maxnature_maxlearner_randompicks[agent]/Eval_MultiGrid-SixteenRooms-v0/solved_rate/out.csv", 3)