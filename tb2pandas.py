# inspired by https://laszukdawid.com/blog/2021/01/26/parsing-tensorboard-data-locally/

import os
import pandas as pd
from tensorflow.python.summary.summary_iterator import summary_iterator
from tqdm import tqdm


def convert_tb_data(root_dir, sort_by=None):
    """Convert local TensorBoard data into Pandas DataFrame.

    Function takes the root directory path and recursively parses
    all events data.
    If the `sort_by` value is provided then it will use that column
    to sort values; typically `wall_time` or `step`.

    *Note* that the whole data is converted into a DataFrame.
    Depending on the data size this might take a while. If it takes
    too long then narrow it to some sub-directories.

    Paramters:
        root_dir: (str) path to root dir with tensorboard data.
        sort_by: (optional str) column name to sort by.

    Returns:
        pandas.DataFrame with [wall_time, name, step, value] columns.

    """

    def convert_tfevent(filepath):
        return pd.DataFrame([
            parse_tfevent(e) for e in summary_iterator(filepath) if len(e.summary.value)
        ])

    def parse_tfevent(tfevent):
        return dict(
            wall_time=tfevent.wall_time,
            name=tfevent.summary.value[0].tag,
            step=tfevent.step,
            value=float(tfevent.summary.value[0].simple_value),
        )

    columns_order = ['wall_time', 'name', 'step', 'value']

    out = []
    for (root, _, filenames) in os.walk(root_dir):
        for filename in filenames:
            if "events.out.tfevents" not in filename:
                continue
            file_full_path = os.path.join(root, filename)
            out.append(convert_tfevent(file_full_path))

    # Concatenate (and sort) all partial individual dataframes
    all_df = pd.concat(out)[columns_order]
    if sort_by is not None:
        all_df = all_df.sort_values(sort_by)

    return all_df.reset_index(drop=True)

def split_tbdf_by_expname(df):
    all_names = pd.unique(df['name'])
    splits_by_identifier = {}
    for name in all_names:
        named_df = df[df['name'] == name]
        split_name = name.split("/")
        identifier = split_name[0]
        plot_name = "/".join(split_name[1:])

        if identifier not in splits_by_identifier:
            splits_by_identifier[identifier] = {}

        splits_by_identifier[identifier][plot_name] = named_df
    return splits_by_identifier


if __name__ == "__main__":
    dir_path = "/home/velythyl/Desktop/diayn-coop-ued/tb_extract"

    experiments_names = os.listdir(dir_path)
    experiment_paths = map(lambda x: f"{dir_path}/{x}", experiments_names)
    experiment_paths = filter(lambda x: os.path.isdir(x), experiment_paths)
    final_experiment_names = map(lambda x: x[len(dir_path):], experiment_paths)

    for path, name in tqdm(zip(experiment_paths, final_experiment_names)):
        df = convert_tb_data(path, sort_by='step')
        splitted = split_tbdf_by_expname(df)

        out_path = f"./out/{name}"

        for key in splitted:
            for subkey in splitted[key]:
                splitted_path = f"{out_path}/{key}/{subkey}"
                os.makedirs(splitted_path)
                splitted[key][subkey].to_csv(splitted_path+"/out.csv")
        df.to_csv(f"{out_path}/out.csv")
