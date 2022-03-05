import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
import re

from utils import run

SMOOTHINGS = [0,
              #0.5, 0.6, 0.7, 0.8,
              0.9, 0.95, 0.59, 0.69, 0.79, 0.89]
DEBUG = False
OUTDIR = "img_out"
EWM = True

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
                if str(smoothing).endswith("9") and "solve" in csv:
                    run = frame.groupby('step').mean()
                    run = run.ewm(alpha=(1-smoothing))

                    y_values = run.mean()['value']
                    y_errors = run.std()['value']
                    print(y_values)
                    x_y_e.append((x_values, y_values, y_errors, name))
                    continue

                acc = []
                for run_id in range(num_runs):
                    run = frame.iloc[run_id::num_runs,:]
                    if smoothing == 0:
                        acc.append(run['value'])
                    else:
                        acc.append(
                            run.ewm(alpha=(1-smoothing)).mean()['value'] if EWM else
                            run.rolling(2).mean()['value']
                        )
                assert len(acc) == num_runs

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
                    fig, axes =init_plot(True)
                    just_one_plot(ax=axes[0,0], data=(x_values, y_values, y_errors, name), exp_name=path)
                    do_finish_plot(fig=fig, csv_name=csv, exp_name=path, smoothing_id=smoothing)
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

def init_plot(solo):
    if solo:
        fig, ax = plt.subplots(1, squeeze=False)
    else:
        fig, ax = plt.subplots(2, 4, squeeze=False, figsize=(6*4, 6*2), sharex=True, sharey=True, dpi=200)
        plt.tight_layout()
        #fig.subplots_adjust(wspace=0.01, hspace=0.01, bottom=0.099999, left=0.07)
    return fig, ax

def get_title_and_axis_name(csv_name):
    if "Eval_" in csv_name:
        plot_name, y_axis_name = csv_name.replace("Eval_", "").replace("MultiGrid-", "").replace("-Minigrid",
                                                                                                 "").replace("-v0",
                                                                                                             "").replace(
            "/out.csv", "").split(
            "/")
    elif "teacher-learner" in csv_name:
        print(csv_name)
        y_axis_name = csv_name.replace("teacher-learner", "").replace("/Teacher/", "").replace("_", " ").replace(
            "/out.csv", "")
        plot_name = y_axis_name
    else:
        plot_name = csv_name
        y_axis_name = plot_name
    plot_name = plot_name.title()

    #plt.title(plot_name)

    y_axis_name = {
        "solved_rate": "Solve Rate",
        "median": "Median",
        "min": "Min",
        "shortestpathlength": "Shortest Path Length",
        "shortestpathlength std": "Shortest Path Length STD",
        "n clutter placed": "Number of Blocks Placed"
    }.get(y_axis_name, y_axis_name)
    if "teacher-learner" in csv_name:
        plot_name = y_axis_name

    return plot_name, y_axis_name

def get_cleaned_exp_name(exp_name, ax=None):

    # old:
    """
    all = {"blocks", "agent", "goal"}

    exp_name = re.search(r"\[(.*)\]", exp_name)
    if exp_name is None:
        exp_name = "all"
    else:
        exp_name = exp_name.group(1)
        if len(exp_name.split(",")) == 3:
            if ax is not None:
                ax.get_legend().remove()
            exp_name = "none"
        else:
            intelligent_powers = all - set(exp_name.split(","))
            exp_name = ",".join(intelligent_powers)
    exp_name = f"picks_{exp_name}".replace(",", "_and_")
    """
    # new:

    scheme = "Cooperative"
    if "adversarial" in exp_name:
        scheme = "Adversarial"
    scheme = f"({scheme})"

    string = []
    if "icm" in exp_name:
        if "mix" in exp_name:
            string.append(f"ICM-Extrinsic Mix")
        else:
            string.append("ICM")
    else:
        string.append(f"Extrinsic")

    other_info = []
    if "bumps" in exp_name:
        other_info.append("NB")

    if "blocks" in exp_name:
        other_info.append("RB")

    return " ".join(string) + " " + (f'({",".join(other_info)})' if len(other_info) > 0 else '')

def do_finish_plot(fig, axes, csv_name, exp_name, smoothing_id, override_path="", just_one=False):
    plot_name, y_axis_name = get_title_and_axis_name(csv_name)
    fig.supxlabel("Timestep")
    fig.supylabel(y_axis_name)

    #plt.ylabel(y_axis_name)
    plt.grid(alpha=0.3)

    exp_name = get_cleaned_exp_name(exp_name)
    if override_path != "":
        exp_name = override_path
    plt.subplots_adjust(top=0.93)
    fig.suptitle(exp_name + "\n")
    exp_name = f"{OUTDIR}/{exp_name}"

    os.makedirs(exp_name, exist_ok=True)

    handles, labels = axes[0,0].get_legend_handles_labels()

    if just_one:
        added_indep_once = False
        for i in range(2):
            for j in range(4):
                handles_temp, labels_temp = axes[i,j].get_legend_handles_labels()
                if len(labels_temp) > 0 and labels_temp[0] == "independent" and not added_indep_once:
                    handles.append(handles_temp[0])
                    labels.append(labels_temp[0])
                    added_indep_once = True
                if len(labels_temp) > len(labels):   # todo temp, remove
                    handles = handles_temp
                    labels = labels_temp

                leg = axes[i, j].get_legend()
                if leg is not None:
                    leg.remove()
        if "clutter" not in csv_name:
            fig.legend(handles, labels, loc='lower center', fontsize=20, bbox_to_anchor=(0.38, 0.26))
        else:
            fig.legend(handles, labels, loc='lower center', fontsize=20, bbox_to_anchor=(0.38, 0.12))
    # todo temp png to pdf
    plt.savefig(f"{exp_name}/{plot_name}_{y_axis_name}_{SMOOTHINGS[smoothing_id]}.pdf", dpi=fig.dpi)
    if DEBUG:
        plt.show()
    plt.close(fig)

def just_one_plot(ax, data, exp_name, just_one=False):
    x_values, y_values, y_errors, csv_name = data

    line_label = ""
    if "cooperative" in exp_name:
        line_label = "cooperative"
    if "adversarial" in exp_name:
        line_label = "adversarial"

    my_cmap = sns.color_palette("colorblind", as_cmap=True)
    color = my_cmap[0] if line_label == "cooperative" else my_cmap[1]

    plot_name = get_cleaned_exp_name(exp_name)
    if "icm" in plot_name.lower() and "mix" not in plot_name.lower():
        color = my_cmap[6]
        temp_line_label = "independent"
        sns.lineplot(ax=ax, x=x_values, y=y_values, label=temp_line_label, color=color)
    else:
        sns.lineplot(ax=ax, x=x_values, y=y_values, label=line_label, color=color)

    past_work = {
        "maze": {
            "paired": (0,0),
            #"repaired": (0.2, 0.1),
            #"plr": (0.3, 0.1),
            "plr⊥": (0.6, 0.1),
            #"plr⊥ (500M)": (0.5, 0.1)
        },
        "sixteen": {
            "paired": (0.7,0.1),
            #"repaired": (0.9, 0.1),
            #"plr": (1, 0.0),
            "plr⊥": (0.8, 0.1),
            #"plr⊥ (500M)": (1, 0.0)
        },
        "laby": {
            "paired": (0.3,0.1),
            #"repaired": (0.1, 0.0),
            #"plr": (0.3, 0.1),
            "plr⊥": (0.5, 0.1),
            #"plr⊥ (500M)": (0.7, 0.1)
        }
    }


    x_ticks = ((1+np.arange(30)) * 100000)[1::2]
    x_labels = list(map(lambda x: f"{str(x / 1000000)[:3]}M", x_ticks))

    #plt.locator_params(axis='y', nbins=10)

    if "solve" in csv_name:
        ax.set_ylim([0, 1.1])
        y_ticks = np.arange(11) / 10
        ax.set_yticks(y_ticks, y_ticks, fontsize=14)    # was 20
    elif "path" in csv_name:
        ax.set_ylim([12.5, 32.5])

        steps = 1 + (30-12.5) // 2.5

        y_ticks = np.arange(steps) * 2.5 + 12.5
        ax.set_yticks(y_ticks, y_ticks, fontsize=14)    # was 20
    elif "clutter" in csv_name:
        LOW = 37.5
        HIG = 43.5
        STEP = 0.5
        ax.set_ylim([LOW, HIG+STEP])

        steps = 1 + abs((LOW-HIG) // STEP)

        y_ticks = np.arange(steps) * STEP + LOW
        ax.set_yticks(y_ticks, y_ticks, fontsize=14)    # was 20
    ax.set_xticks(x_ticks, x_labels, ha='right', rotation=45, rotation_mode="anchor", fontsize=14)
    # ax.set_xticklabels(list(map(str, x_values)))
    ax.fill_between(x_values, y_values - y_errors, y_values + y_errors, alpha=0.25, color=color)


    if line_label == "cooperative":
        doable = False
        for key in past_work.keys():
            if key in csv_name.lower():
                doable = True

        if "nb" in plot_name.lower() and doable:
            style = "--"
            color = my_cmap[2]

            ax.axhline(0.005, linestyle=style, color=color, label="paired".upper())
            # ax.fill_between(x_values, old_work[0])
        else:
            for key in past_work.keys():
                if key in csv_name.lower():
                    for old_work, (val, std) in past_work[key].items():
                        style = "--" if old_work == "paired" else "dashdot"
                        color = my_cmap[2] if old_work == "paired" else my_cmap[4]

                        if val == 0:
                            val = 0.005

                        ax.axhline(val, linestyle=style, color=color, label=old_work.upper())
                        #ax.fill_between(x_values, old_work[0])

    #plot_name, y_axis_name = get_title_and_axis_name(csv_name)

    plot_name = plot_name.replace("_", " ").title()
    plot_name = plot_name.replace("Nb", "NB").replace("Rb", "RB").replace("Icm", "ICM")

    if not just_one:
        ax.set_title(plot_name, fontsize=22, pad=-22)

    get_cleaned_exp_name(exp_name, ax)


    ax.set_ylabel('')


def plot(data, keys, names):
    # solo plots


    for key in keys:


        #try:
        for smoothing_id in range(len(SMOOTHINGS)):
            fig, axes = init_plot(True)
            for dict_id, dict in enumerate(data):
                packing = dict[key]
                just_one_plot(data=packing[smoothing_id], exp_name=names[dict_id], ax=axes[0,0])
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

    num_run_matchings = dict()
    path_matchings = dict()
    for sub_paths in experiment_names:
        pattern = sub_paths.replace("cooperative", "PATTERN").replace("adversarial", "PATTERN")
        if pattern not in path_matchings:
            path_matchings[pattern] = []
        path_matchings[pattern].append(sub_paths)

        df = pd.read_csv(f"{path}/{sub_paths}/Eval_MultiGrid-SixteenRooms-v0/solved_rate/out.csv")
        df = df[df['step'] == 100000]
        num_seeds = len(df)
        num_run_matchings[sub_paths] = num_seeds

    pairs = []
    pairs_nb = []
    for key in path_matchings.keys():
        pair = sorted(path_matchings[key])
        if "bumps" in pair[0]:
            pairs_nb.append(pair)
        else:
            pairs.append(pair)
    pairs = sorted(pairs, key=lambda x: x[0])
    pairs_nb = sorted(pairs_nb, key=lambda x: x[0])
    # todo
    #for key, values in path_matchings.items():
    #    assert len(values) == 2 or "blocks,goal,agent" in values[0]

    experiment_names = []
    num_runs = []
    for pair in (pairs + pairs_nb):
        experiment_names.append(list(map(lambda x: f"{path}/{x}", pair)))

        nums = list(map(lambda x: num_run_matchings[x], pair))
        num_runs.append(nums)

    return experiment_names, num_runs

def filter_keys(common_keys):
    keys = []
    for key in common_keys:
        if "solve" in key or "path" in key or "clutter" in key:
            pass
        else:
            continue

        if "Eval" in key or "teacher-learner" in key:
            pass
        else:
            continue
        keys.append(key)
    return set(keys)

def find_all_experiment_paths(path):
    all_paths = run(f"find {path} -name 'out.csv'").split("\n")
    csv_paths = set(map(lambda x: "/".join(x.replace(path, "").split("/")[2:]), all_paths))
    return csv_paths

def do_mp(pair, csv_paths, num_runs):
    print("PAIR")
    print(pair)

    result = [do_csv(pair[0], csv_paths, num_runs[0])]
    keys1 = set(result[0].keys())
    if len(pair) == 2:
        result.append(do_csv(pair[1], csv_paths, num_runs[1]))
        keys2 = set(result[1].keys())
        uh = keys1 == keys2  # if we dont do this, sets are counted as empty... fuck python
        common_keys = keys1.intersection(keys2)
    else:
        common_keys = keys1
    common_keys = filter_keys(common_keys)

    #plot(result, common_keys, pair)

    return result, common_keys, pair


if __name__ == "__main__":
    path = "/home/velythyl/Desktop/tb-extract/out"

    sns.set()
    sns.set_palette("colorblind")
    matplotlib.rcParams.update({'font.size': 14})   # was 18
    csv_paths = filter_keys(find_all_experiment_paths(path))
    exp_pairs, num_runs = find_all_experiment_pairs(path)

    if DEBUG:
        do_mp(exp_pairs[0], csv_paths)
        do_csv(exp_pairs[0][0], csv_paths, 3)

    import multiprocessing as mp
    pool = mp.Pool()
    results = pool.starmap(do_mp, zip(exp_pairs, [csv_paths] * len(exp_pairs), num_runs))
    pool.close()
    pool.join()

    temp_common_keys = []
    for (_, common_keys, _) in results:
        temp_common_keys.append(common_keys)
    all_common_keys = temp_common_keys[0]
    for temp_common_key in temp_common_keys:
        all_common_keys = all_common_keys.intersection(temp_common_key)
    #all_common_keys = filter_keys(all_common_keys)

    #nb_indices = []
    #other_indices = []
    #for i, pair in enumerate(exp_pairs):
    #    if
    #indices = np.argsort(np.array(list(
    #    map(lambda x: ("NB"+x) if "nb" in x.lower() else x,
    #        map(lambda xy: ":".join(xy), exp_pairs)
    #    )
    #)))
    indices = np.arange(len(results))


    for smoothing_id in range(len(SMOOTHINGS)):
        for key in all_common_keys:
            #if "Eval" and "solve" not in key:
            #    continue
            """
            fig, axes = init_plot(False)
            for index in indices:
                (dicts, _, name_pair) = results[index]
                for dict_id, dict in enumerate(dicts):
                    packing = dict[key]

                    x_axes = int(index >= 4)
                    y_axes = index % 4
                    just_one_plot(ax=axes[x_axes, y_axes], data=packing[smoothing_id], exp_name=name_pair[dict_id])

            title = get_title_and_axis_name(key)[0]
            title = title.replace("_", " ").title()
            do_finish_plot(override_path=get_title_and_axis_name(key)[0], fig=fig, csv_name=key, exp_name=exp_pairs[index][0], smoothing_id=smoothing_id, axes=axes)
            """

            for index in indices:
                (dicts, _, name_pair) = results[index]

                if not ("bumps" and "blocks" in name_pair[0]):
                    continue

                if not ("maze" or "laby" or "sixte" in key.lower()):
                    continue

                fig, axes = init_plot(True)
                for dict_id, dict in enumerate(dicts):
                    packing = dict[key]

                    just_one_plot(ax=axes[0,0], data=packing[smoothing_id], exp_name=name_pair[dict_id], just_one=True)

                title = get_title_and_axis_name(key)[0]
                title = title.replace("_", " ").title()
                do_finish_plot(override_path=get_title_and_axis_name(key)[0], fig=fig, csv_name=key, exp_name=exp_pairs[index][0], smoothing_id=smoothing_id, axes=axes)
            #plt.show()
            #exit()

"""
    key2results_dict = {}
    for key in all_common_keys:
        for (_, dicts, pairs) in results:
            if key not in key2results_dict:
                key2results_dict = []
            key2results_dict[tuple(pairs)] =
"""
   # do_csv("/home/velythyl/Desktop/tb-extract/out/none_none_MinigridSettable-v0_cooperative_maxnature_maxlearner_randompicks[agent]/Eval_MultiGrid-SixteenRooms-v0/solved_rate/out.csv", 3)