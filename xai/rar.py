import os

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from ThirdEye.ase22.config import Config
import pandas as pd

#taking values from csv and recompute the rar value
def compute_road_percentage(cfg, simulation_name, heatmaptype):
    path = os.path.join(cfg.TESTING_DATA_DIR,  # "C:/Users/Linfe/Downloads/data-ASE2022/benchmark-ASE2022"
                        simulation_name,
                        'segmentation-' + heatmaptype,
                        'driving_log.csv')
    if "\\\\" in path:
        path = path.replace("\\\\", "/")
    elif "\\" in path:
        path = path.replace("\\", "/")
    data_df = pd.read_csv(path)

    road_attention = data_df["road_attention"]
    avg_road_attention = data_df["avg_road_attention"]
    all_attention = data_df["all_attention"]
    avg_all_attention = data_df["avg_all_attention"]
    list_percentage = []

    for idx, val in tqdm(enumerate(avg_road_attention)):
        if(avg_road_attention.get(idx) == 0):
            road_pixel = 0
        else:
            road_pixel = road_attention.get(idx)/avg_road_attention.get(idx)
        total_pixel = all_attention.get(idx)/avg_all_attention.get(idx)

        result = road_pixel/total_pixel

        list_percentage.append(result)

    file_name = "htm-" + heatmaptype.lower() + '-scores'
    path_name = os.path.join(cfg.TESTING_DATA_DIR,
                             simulation_name,
                             file_name + '-road_percentage')

    np.save(path_name, list_percentage)

    plt.hist(list_percentage)
    plt.title("avg_road_attention_percentage")
    path_name = os.path.join(cfg.TESTING_DATA_DIR,
                             simulation_name,
                             'plot-' + file_name + '-road_percentage.png')
    plt.savefig(path_name)
    plt.close()


if __name__ == '__main__':
    cfg = Config()
    cfg.from_pyfile(filename="/mnt/c/Unet/ThirdEye/ase22/config_my.py")

    for condition in ['mutants', 'ood', 'icse20', ]:
        condition_path = os.path.join(cfg.TESTING_DATA_DIR, condition)
        condition_files = os.listdir(condition_path)
        for sim in condition_files:
            for attention_type in ["Faster-ScoreCAM"]:
                print(f"heatmap: {attention_type}, simulation_name: {condition + '/' + sim}")
                compute_road_percentage(cfg, simulation_name=condition + '/' + sim,
                                        heatmaptype=attention_type,
                                        )

    for condition in ['gauss-journal-track1-nominal']:
        for attention_type in ["Faster-ScoreCAM"]:
            condition_path = os.path.join(cfg.TESTING_DATA_DIR, condition)
            condition_files = os.listdir(condition_path)
            # for sim in condition_files:
            print(f"heatmap: {attention_type}, simulation_name: {condition}")
            compute_road_percentage(cfg, simulation_name=condition,  # condition + '/' + sim,
                                    heatmaptype=attention_type,
                                    )
