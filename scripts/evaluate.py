import scripts.evaluate_failure_prediction_heatmaps_scores as eva
from utils import *
import scripts.evaluate_failure_prediction_seg_merge_hm as eva_new

# (cfg, heatmap_type, simulation_name, summary_type, aggregation_method, condition)

if __name__ == '__main__':

    cfg = Config()
    # cfg.from_pyfile(filename="/mnt/c/Unet/ThirdEye/ase22/config_my.py")  # ""
    cfg.from_pyfile(filename="C:/Unet/ThirdEye/ase22/config_my.py")

    # For ThirdEye set segmentation to false
    segmentation = True
    for ht in ['smoothgrad']:  # for normal remove faster-scorecam,,'faster-scorecam'
        for condition in ['mutants', 'ood', 'icse20']:  # ,,
            condition_path = os.path.join(cfg.TESTING_DATA_DIR, condition)
            condition_files = os.listdir(condition_path)
            for simulation_name in condition_files:
                for st in ['-avg_road_attention_percentage'
                           ]:  # '-avg','-avg-grad','-road_percentage',,'-total_road_attention_percentage'
                    for am in ['max']:#'mean',
                        print(condition + '/' + simulation_name)
                        eva.evaluate_failure_prediction(cfg=cfg, heatmap_type=ht,
                                                            simulation_name=condition + '/' + simulation_name,
                                                            summary_type=st, aggregation_method=am, condition=condition,
                                                            segmentation=segmentation)  # mutants
                        '''
                        for the union version 
                        eva_new.evaluate_failure_prediction(cfg=cfg, heatmap_type=ht,
                                                        simulation_name=condition + '/' + simulation_name,
                                                        summary_type=st, aggregation_method=am, condition=condition,
                                                        segmentation=segmentation)  # mutants
                        '''