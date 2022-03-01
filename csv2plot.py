import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
import re

from utils import run

SMOOTHINGS = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
DEBUG = False
OUTDIR = "img_out"

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
            for smoothing_id, smoothing in enumerate(SMOOTHINGS):
                acc = []
                for run_id in range(num_runs):
                    run = frame.iloc[run_id::3,:]
                    if smoothing == 0:
                        acc.append(run['value'])
                    else:
                        acc.append(run.ewm(alpha=(1-smoothing)).mean()['value'])

                min_len = min(map(len, acc))
                acc = list(map(lambda ac: ac[:min_len], acc))

                if min_len > 35:
                    x_values = np.arange(min_len) * (3000000 // min_len)

                acc = [ac.to_numpy() for ac in acc]
                acc = np.array(acc).astype(float)
                y_values = acc.mean(axis=0)
                y_errors = acc.std(axis=0)

                if DEBUG:
                    if "solve" not in csv:
                        continue
                    fig, axes = plt.figure(figsize=(6,6))
                    just_one_plot(axes=axes, data=(x_values, y_values, y_errors, name), exp_name=path, subplot_y=0, subplot_x=0)
                    do_finish_plot(fig=fig, csv_name=csv, exp_name=path, smoothing_id=smoothing)
                    plt.show()
                    exit()

                x_y_e.append((x_values, y_values, y_errors, name))
            csv_dict[csv] = x_y_e

        except ValueError as e:
            print(csv)
            raise e
            if 'setting an array element with a sequence' in e.args[0]:
                assert "teacher" or "last_step" in csv.lower()
            else:
                print(f"WARNING: assumed CSV {csv} was invalid and skipped it.")
                raise e

    return csv_dict

def do_finish_plot(fig, csv_name, exp_name, smoothing_id):
    plt.xlabel("Timestep")

    if "Eval_" in csv_name:
        plot_name, y_axis_name = csv_name.replace("Eval_", "").replace("MultiGrid-", "").replace("-Minigrid",
                                                                                                 "").replace("-v0",
                                                                                                             "").replace("/out.csv", "").split(
            "/")
    elif "teacher-learner" in csv_name:
        print(csv_name)
        y_axis_name = csv_name.replace("teacher-learner_", "").replace("/Teacher/", "").replace("_", " ").replace("/out.csv", "")
        plot_name = y_axis_name
    else:
        plot_name = csv_name
        y_axis_name = plot_name
    plot_name = plot_name.title()

    plt.title(plot_name)

    y_axis_name = {
        "solved_rate": "Solve Rate",
        "median": "Median",
        "min": "Min",
        "shortestpathlength": "Shortest Path Length"
    }.get(y_axis_name, y_axis_name)

    plt.ylabel(y_axis_name)
    plt.tight_layout()
    plt.grid(alpha=0.3)

    exp_name = re.search(r"\[(.*)\]", exp_name)
    if exp_name is None:
        exp_name = "none"
    else:
        exp_name = exp_name.group(1)
        if len(exp_name.split(",")) == 3:
            ax = plt.gca()
            ax.get_legend().remove()
    exp_name = f"{OUTDIR}/random_picks_{exp_name}".replace(",", "_and_")

    os.makedirs(exp_name, exist_ok=True)

    plt.savefig(f"{exp_name}/{plot_name}_{y_axis_name}_{SMOOTHINGS[smoothing_id]}.png", dpi=fig.dpi)
    plt.close(fig)

def just_one_plot(data, exp_name):
    x_values, y_values, y_errors, csv_name = data

    line_label = ""
    if "cooperative" in exp_name:
        line_label = "cooperative"
    if "adversarial" in exp_name:
        line_label = "adversarial"

    ax = sns.lineplot(x=x_values, y=y_values, label=line_label)

    x_ticks = ((1+np.arange(30)) * 100000)[1::2]
    x_labels = list(map(lambda x: f"{str(x / 1000000)[:3]}M", x_ticks))

    #plt.locator_params(axis='y', nbins=10)

    if "solve" in exp_name:
        ax.set_ylim([0, 1.1])
        y_ticks = np.arange(11) / 10
        ax.set_yticks(y_ticks, y_ticks)
    ax.set_xticks(x_ticks, x_labels, ha='right', rotation=45, rotation_mode="anchor")
    # ax.set_xticklabels(list(map(str, x_values)))
    ax.fill_between(x_values, y_values - y_errors, y_values + y_errors, alpha=0.5)


def plot(data, keys, names):
    # solo plots


    for key in keys:

        if "solve" in key or "path" in key:
            pass
        else:
            continue

        #try:
        for smoothing_id in range(len(SMOOTHINGS)):
            fig = plt.figure(figsize=(6,6))
            for dict_id, dict in enumerate(data):
                packing = dict[key]
                just_one_plot(data=packing[smoothing_id], exp_name=names[dict_id])
            do_finish_plot(fig=fig, csv_name=key, exp_name=names[0], smoothing_id=smoothing_id)
        #plt.show()
        #exit()
        #except Exception as e:
        #    pass

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
    keys1 = set(result[0].keys())
    if len(pair) == 2:
        result.append(do_csv(pair[1], csv_paths, 3))
        keys2 = set(result[1].keys())
        uh = keys1 == keys2  # if we dont do this, sets are counted as empty... fuck python
        common_keys = keys1.intersection(keys2)
    else:
        common_keys = keys1

    plot(result, common_keys, pair)

    return result, common_keys, pair


if __name__ == "__main__":
    path = "/home/velythyl/Desktop/tb-extract/out_temp"

    sns.set()
    sns.set_palette("colorblind")
    matplotlib.rcParams.update({'font.size': 14})
    csv_paths = find_all_experiment_paths(path)
    exp_pairs = find_all_experiment_pairs(path)

    if DEBUG:
        do_mp(exp_pairs[0], csv_paths)
        do_csv(exp_pairs[0][0], csv_paths, 3)

    import multiprocessing as mp
    pool = mp.Pool()
    results = pool.starmap(do_mp, zip(exp_pairs, [csv_paths] * len(exp_pairs)))
    pool.close()
    pool.join()

    temp_common_keys = []
    for (_, common_keys, _) in results:
        temp_common_keys.append(common_keys)
    all_common_keys = temp_common_keys[0]
    for temp_common_key in temp_common_keys:
        all_common_keys = all_common_keys.intersection(temp_common_key)

    indices = np.argsort(np.array(list(map(lambda xy: xy[0], exp_pairs))))

"""
    for smoothing_id in range(len(SMOOTHINGS)):
        for key in all_common_keys:
            #if "Eval" and "solve" not in key:
            #    continue
            fig, axes = init_plot(False)
            for index in indices:
                for dict_id, dict in enumerate(results[index]):
                    packing = dict[key]
                    just_one_plot(fig=fig, data=packing[smoothing_id], exp_name=exp_pairs[index][dict_id], subplot_x=index % 4, subplot_y=in)
            do_finish_plot(fig=fig, csv_name=key, exp_name=names[0], smoothing_id=smoothing_id)
            #plt.show()
            #exit()
"""
"""
    key2results_dict = {}
    for key in all_common_keys:
        for (_, dicts, pairs) in results:
            if key not in key2results_dict:
                key2results_dict = []
            key2results_dict[tuple(pairs)] =
"""
   # do_csv("/home/velythyl/Desktop/tb-extract/out/none_none_MinigridSettable-v0_cooperative_maxnature_maxlearner_randompicks[agent]/Eval_MultiGrid-SixteenRooms-v0/solved_rate/out.csv", 3)