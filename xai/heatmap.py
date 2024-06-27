import matplotlib.pyplot as plt
import tensorflow.python.keras.models
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.scorecam import Scorecam
from tf_keras_vis.utils.scores import CategoricalScore
from tqdm import tqdm

import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import torch

import torchvision.transforms.functional as TF
from torchvision import transforms
import cv2 as cv2



import shutil as shutil
import os
import utils as utils
from config import *
from U_Net import U_Net

import pandas as pd
def preprocessForSegmentation(img):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    '''
    This turning the input image into Unet model format
    '''
    # bei bedarf diese Zeile auskommentieren
    # img = img[:, :, :3]
    image = TF.to_tensor(img)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    image = normalize(image)
    image = image.to(device)
    image = image.unsqueeze(0)
    return image


def merge(saliency_map, predicted_rgb):
    saliency_map_seg = saliency_map
    all_attention = road_attention = road_pixel = all_pixel = 0
    for x in range(saliency_map.shape[0]):
        for y in range(saliency_map.shape[1]):
            all_attention += saliency_map_seg[x, y]
            ####################
            all_pixel += 1
            ####################
            if np.all(predicted_rgb[x, y] == [0, 255, 0]):
                saliency_map_seg[x, y] = 0
            else:
                road_pixel += 1
                road_attention += saliency_map[x, y]

    # In case of ads is crashed with no road part in img
    if road_pixel == 0:
        avg_road_attention = 0
    else:
        avg_road_attention = road_attention / road_pixel

    avg_all_attention = all_attention / all_pixel

    return saliency_map_seg, avg_road_attention, avg_all_attention, road_attention, all_attention


mapping = {
    149: 0,
    29: 1
}
mappingrgb = {
    29: (0, 0, 255),
    149: (0, 255, 0)
}


def class_to_rgb(mask):
    '''
    This function maps the classification index ids into the rgb.
    For example after the argmax from the network, you want to find what class
    a given pixel belongs too. This does that but just changes the color
    so that we can compare it directly to the rgb groundtruth label.
    '''
    mask2class = dict((v, k) for k, v in mapping.items())
    rgbimg = torch.zeros((3, mask.size()[0], mask.size()[1]), dtype=torch.uint8)
    for k in mask2class:
        rgbimg[0][mask == k] = mappingrgb[mask2class[k]][0]
        rgbimg[1][mask == k] = mappingrgb[mask2class[k]][1]
        rgbimg[2][mask == k] = mappingrgb[mask2class[k]][2]
    return rgbimg


def score_when_decrease(output):
    # return -1.0 * output[:, 0]
    return CategoricalScore(0)


def compute_heatmap(cfg, simulation_name, crop, if_resize, yuv, input_model, condition, attention_type="SmoothGrad"):
    # prepare segmentation model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = U_Net(3, 2)
    model.to(device)

    checkpoint_path = '/home/xchen/Documents/linfeng/BA-XAI/SegmentationModel_CrossEntropyLoss38.pth'
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    """
    Given a simulation by Udacity, the script reads the corresponding image paths from the csv and creates a heatmap for
    each driving image. The heatmap is created with the SmoothGrad algorithm available from tf-keras-vis
    (https://keisen.github.io/tf-keras-vis-docs/examples/attentions.html#SmoothGrad). The scripts generates a separate
    IMG/ folder and csv file.
    """

    print("Computing attention heatmaps for simulation %s using %s" % (simulation_name, attention_type))

    # load the image file paths from csv

    path = os.path.join(cfg.TESTING_DATA_DIR,  # "/mnt/c/Unet/benchmark-ASE2022/"
                        simulation_name,  # gauss-journal-track1-nominal
                        'driving_log.csv')

    data_df = pd.read_csv(path)
    if condition == 'ood':
        data = data_df["center"].apply(
            lambda
                x: "/home/xchen/Documents/linfeng/BA-XAI/dataset/data-ASE2022/data-ASE2022/benchmark-ASE2022/benchmark-ASE2022/ood" + x.replace(
                "simulations", ""))
    # data = data_df["center"]
    else:
        data = data_df["center"].apply(
            lambda
                x: "/home/xchen/Documents/linfeng/BA-XAI/dataset/data-ASE2022/data-ASE2022/benchmark-ASE2022/benchmark-ASE2022" + x.replace(
                "simulations", ""))
    print("read %d images from file" % len(data))

    #########################################################
    # load self-driving car model
    # path = cfg.SDC_MODELS_DIR + '/' + input_model
    # input_model = input_model[:-3]
    # /home/xchen/Documents/linfeng/BA-XAI/dataset/model
    path = '/home/xchen/Documents/linfeng/BA-XAI/dataset/model/' + '/' + input_model
    self_driving_car_model = tensorflow.keras.models.load_model(
        path)
    #########################################################
    # load attention model
    saliency = None
    # attention_type: ["SmoothGrad", "GradCam++"]
    if attention_type == "SmoothGrad":
        saliency = Saliency(self_driving_car_model, model_modifier=None)
    elif attention_type == "GradCam++":
        saliency = GradcamPlusPlus(self_driving_car_model, model_modifier=None)
    elif attention_type == "Faster-ScoreCAM":
        saliency = Scorecam(self_driving_car_model, model_modifier=None)
    avg_heatmaps = []
    avg_gradient_heatmaps = []
    list_of_image_paths = []
    total_time = 0
    prev_hm = gradient = gradient_seg = prev_hm_seg = np.zeros((80, 160))

    # make same copy for with segmentation
    avg_gradient_heatmaps_seg = []
    list_of_image_paths_seg = []
    total_time = 0
    prev_hm = gradient = gradient_seg = prev_hm_seg = np.zeros((80, 160))
    list_of_total_road_attention_percentage = []
    list_of_avg_road_attention_percentage = []
    list_of_road_attention = []
    list_of_all_attention = []
    list_of_avg_road_attention = []
    list_of_avg_all_attention = []

    # create directory for the heatmaps
    path_save_heatmaps = os.path.join(cfg.TESTING_DATA_DIR,
                                      simulation_name,
                                      "heatmaps-" + attention_type.lower(),
                                      "IMG")

    if os.path.exists(path_save_heatmaps):
        print("Deleting folder at {}".format(path_save_heatmaps))
        shutil.rmtree(path_save_heatmaps)
    print("Creating image folder at {}".format(path_save_heatmaps))
    os.makedirs(path_save_heatmaps)

    path_save_heatmaps_seg = os.path.join(cfg.TESTING_DATA_DIR,
                                          simulation_name,
                                          'segmentation-' + attention_type,
                                          "IMG")

    if os.path.exists(path_save_heatmaps_seg):
        print("Deleting folder at {}".format(path_save_heatmaps_seg))
        shutil.rmtree(path_save_heatmaps_seg)
    print("Creating image folder at {}".format(path_save_heatmaps_seg))
    os.makedirs(path_save_heatmaps_seg)

    # i did not safe the heatmap only

    for idx, img in tqdm(enumerate(data)):
        # img = "/mnt/c/Unet/dataset5" + img

        # img = '/mnt/c/Unet/benchmark-ASE2022/mutants/udacity_add_weights_regularisation_mutated0_MP_l1_3_1/IMG/2022_04_21_13_11_45_057.jpg'
        # convert Windows path, if needed
        if "\\\\" in img:
            img = img.replace("\\\\", "/")
        elif "\\" in img:
            img = img.replace("\\", "/")

        # load image: x is for heatmap and y for segmentation
        x = y = image = mpimg.imread(img)

        y = preprocessForSegmentation(y)

        with torch.no_grad():
            prediction = model(y)

        predicted_rgb = torch.zeros((3, prediction.size()[2], prediction.size()[3])).to(device)
        maxindex = torch.argmax(prediction[0], dim=0).cpu().int()
        predicted_rgb = class_to_rgb(maxindex).to('cpu')
        predicted_rgb = predicted_rgb.squeeze().permute(1, 2, 0).numpy()

        if crop:
            x = utils.crop(x)
            predicted_rgb = utils.crop(predicted_rgb)
        if if_resize:
            x = utils.resize(x)
            predicted_rgb = utils.resize(predicted_rgb)
        else:
            x = cv2.resize(x, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)
            predicted_rgb = cv2.resize(predicted_rgb, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)
        if yuv:
            x = utils.rgb2yuv(x)

        x = x.astype('float32')

        # compute heatmap image
        saliency_map = None
        if attention_type == "SmoothGrad":
            # saliency_map = saliency(score_when_decrease, x, smooth_samples=20, smooth_noise=0.20)
            saliency_map = saliency(CategoricalScore(0), x, smooth_samples=20, smooth_noise=0.20)
        if attention_type == "GradCam++":
            saliency_map = saliency(CategoricalScore(0),
                                    x,
                                    penultimate_layer=-1)
        if attention_type == "Faster-ScoreCAM":
            saliency_map = saliency(CategoricalScore(0),
                                    x,
                                    penultimate_layer=-1,
                                    max_N=20)

        average = np.average(saliency_map)
        # compute gradient of the heatmap
        if idx == 0:
            gradient = 0
        else:
            gradient = abs(prev_hm - saliency_map)
        average_gradient = np.average(gradient)
        prev_hm = saliency_map

        saliency_map = np.squeeze(saliency_map)
        # store the heatmaps
        file_name = img.split('/')[-1]

        file_name = "htm-" + attention_type.lower() + '-' + file_name
        path_name = os.path.join(path_save_heatmaps, file_name)
        # mpimg.imsave(path_name, np.squeeze(saliency_map))

        img_temp = Image.fromarray((saliency_map * 255).astype(np.uint8))
        img_temp.save(path_name)

        list_of_image_paths.append(path_name)

        # merge heatmap and saliency
        saliency_map_seg, avg_road_attention, avg_all_attention, road_attention, all_attention = merge(saliency_map,
                                                                                                       predicted_rgb)

        list_of_total_road_attention_percentage.append(road_attention / all_attention)
        list_of_avg_road_attention_percentage.append(avg_road_attention / avg_all_attention)
        list_of_road_attention.append(road_attention)
        list_of_all_attention.append(all_attention)
        list_of_avg_road_attention.append(avg_road_attention)
        list_of_avg_all_attention.append(avg_all_attention)

        if idx == 0:
            gradient_seg = 0
        else:
            gradient_seg = abs(prev_hm_seg - saliency_map_seg)
        average_gradient_seg = np.average(gradient_seg)
        prev_hm_seg = saliency_map_seg

        '''
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Create a figure with three subplots

        axes[0].imshow(image)
        axes[0].set_title('Original Image')

        axes[1].imshow(saliency_map)
        axes[1].set_title('Saliency Map')

        axes[2].imshow(predicted_rgb)
        axes[2].set_title('SegMask')

        plt.show()

        '''

        # store seg heapmap
        file_name = img.split('/')[-1]
        file_name = "htm-" + attention_type.lower() + '-' + file_name
        path_name = os.path.join(path_save_heatmaps_seg, file_name)

        # TODO only save once
        if attention_type == "Faster-ScoreCAM":
            mpimg.imsave(path_name, predicted_rgb)

        list_of_image_paths_seg.append(path_name)

        avg_heatmaps.append(average)
        avg_gradient_heatmaps.append(average_gradient)
        # store for seg avg
        avg_gradient_heatmaps_seg.append(average_gradient_seg)

    # save scores as numpy arrays
    file_name = "htm-" + attention_type.lower() + '-scores'
    path_name = os.path.join(cfg.TESTING_DATA_DIR,
                             simulation_name,
                             file_name + '-avg')

    np.save(path_name, avg_heatmaps)
    ##################################
    file_name = "htm-" + attention_type.lower() + '-scores'
    path_name = os.path.join(cfg.TESTING_DATA_DIR,
                             simulation_name,
                             file_name + '-total_road_attention_percentage')

    np.save(path_name, list_of_total_road_attention_percentage)

    file_name = "htm-" + attention_type.lower() + '-scores'
    path_name = os.path.join(cfg.TESTING_DATA_DIR,
                             simulation_name,
                             file_name + '-avg_road_attention_percentage')

    np.save(path_name, list_of_avg_road_attention_percentage)
    #################################
    # plot scores as histograms
    plt.hist(avg_heatmaps)
    plt.title("average attention heatmaps")
    path_name = os.path.join(cfg.TESTING_DATA_DIR,
                             simulation_name,
                             'plot-' + file_name + '-avg.png')
    plt.savefig(path_name)
    # plt.show()

    plt.close()
    # seg part
    plt.hist(list_of_total_road_attention_percentage)
    plt.title("total_road_attention_percentage")
    path_name = os.path.join(cfg.TESTING_DATA_DIR,
                             simulation_name,
                             'plot-' + file_name + '-total_road_attention_percentage.png')
    plt.savefig(path_name)

    plt.close()
    # plt.show()
    plt.hist(list_of_avg_road_attention_percentage)
    plt.title("avg_road_attention_percentage")
    path_name = os.path.join(cfg.TESTING_DATA_DIR,
                             simulation_name,
                             'plot-' + file_name + '-avg_road_attention_percentage.png')
    plt.savefig(path_name)

    plt.close()
    #########################################################################################

    path_name = os.path.join(cfg.TESTING_DATA_DIR,
                             simulation_name,
                             file_name + '-avg-grad')
    np.save(path_name, avg_gradient_heatmaps)

    plt.clf()
    plt.hist(avg_gradient_heatmaps)
    plt.title("average gradient attention heatmaps")
    path_name = os.path.join(cfg.TESTING_DATA_DIR,
                             simulation_name,
                             'plot-' + file_name + '-avg-grad.png')
    plt.savefig(path_name)
    plt.close()



    # seg part
    path_name = os.path.join(cfg.TESTING_DATA_DIR,
                             simulation_name,
                             file_name + '-avg-grad_seg')
    np.save(path_name, avg_gradient_heatmaps_seg)


    # save as csv
    df = pd.DataFrame(list_of_image_paths, columns=['center'])
    path = os.path.join(cfg.TESTING_DATA_DIR,
                        simulation_name,
                        'driving_log.csv')
    data_df = pd.read_csv(path)
    if condition == 'icse20':
        data = data_df[["frameId", "crashed"]]
        df['frameId'] = data['frameId'].copy()
        df['crashed'] = data['crashed'].copy()
    else:
        data = data_df[["frameId", "time", "crashed"]]
        # data = data_df[["frameId", "crashed"]]

        # copy frame id, simulation time and crashed information from simulation's csv
        df['frameId'] = data['frameId'].copy()
        df['time'] = data['time'].copy()
        df['crashed'] = data['crashed'].copy()

    # save it as a separate csv
    df.to_csv(os.path.join(cfg.TESTING_DATA_DIR,
                           simulation_name,
                           "heatmaps-" + attention_type.lower(),
                           'driving_log.csv'), index=False)

    # seg part
    df = pd.DataFrame(list_of_image_paths_seg, columns=['center'])
    path = os.path.join(cfg.TESTING_DATA_DIR,
                        simulation_name,
                        'driving_log.csv')
    data_df = pd.read_csv(path)
    if condition == 'icse20':
        data = data_df[["frameId", "crashed"]]
        df['frameId'] = data['frameId'].copy()
        df['crashed'] = data['crashed'].copy()
    else:
        data = data_df[["frameId", "time", "crashed"]]
        # data = data_df[["frameId", "crashed"]]

        # copy frame id, simulation time and crashed information from simulation's csv
        df['frameId'] = data['frameId'].copy()
        df['time'] = data['time'].copy()
        df['crashed'] = data['crashed'].copy()

    df['total_road_attention_percentage'] = list_of_total_road_attention_percentage
    df['avg_road_attention_percentage'] = list_of_avg_road_attention_percentage
    df['road_attention'] = list_of_road_attention
    df['all_attention'] = list_of_all_attention
    df['avg_road_attention'] = list_of_avg_road_attention
    df['avg_all_attention'] = list_of_avg_all_attention

    # save it as a separate csv
    df.to_csv(os.path.join(cfg.TESTING_DATA_DIR,
                           simulation_name,
                           'segmentation-' + attention_type,
                           'driving_log.csv'), index=False)


if __name__ == '__main__':
    print(os.getcwd())

    cfg = Config()
    cfg.from_pyfile(filename="./third_eye/config_my.py")

    path = '/home/xchen/Documents/linfeng/BA-XAI/dataset/dave2_models/models.csv'

    model_df = pd.read_csv(path)
    simulation = model_df["model"]
    if_crop = model_df["if_crop"]
    if_resize = model_df["if_half_size"]
    if_yue = model_df["if_yuv"]
    input_model = model_df["treated_as"]
    counter = 0

    '''
    for condition in ['gauss-journal-track1-nominal']:  
        for attention_type in ["Faster-ScoreCAM", "SmoothGrad"]:
            condition_path = os.path.join(cfg.TESTING_DATA_DIR, condition)
            condition_files = os.listdir(condition_path)
            #for sim in condition_files:
            print(f"heatmap: {attention_type}, simulation_name: {condition}")
            compute_heatmap(cfg, simulation_name=condition, #condition + '/' + sim, 
                                crop=False, 
                                if_resize=True,
                                yuv=True,
                                input_model='track1-dave2-uncropped-mc-034', 
                                condition=condition, 
                                attention_type=attention_type)


    for simulation_name in simulation:
        for attention_type in ["Faster-ScoreCAM", "SmoothGrad"]: #"SmoothGrad",  "Faster-ScoreCAM", "GradCam++"
            if 'mutated0' in simulation_name:
                print(f"heatmap: {attention_type}, simulation_name: {simulation_name}, if_crop {if_crop.get(counter)}, if_resize {if_resize.get(counter)}, if_yue {if_yue.get(counter)}")
                # ["SmoothGrad", "GradCam++"]
                compute_heatmap(cfg, 'mutants/' + simulation_name, if_crop.get(counter), if_resize.get(counter), if_yue.get(counter),
                            input_model.get(counter)[:-3], '', attention_type=attention_type)

        counter += 1'''

    for condition in ['ood', 'icse20']:
        condition_path = os.path.join(cfg.TESTING_DATA_DIR, condition)
        condition_files = os.listdir(condition_path)
        for sim in condition_files:
            for attention_type in ["Faster-ScoreCAM", "SmoothGrad"]:
                print(f"heatmap: {attention_type}, simulation_name: {condition + '/' + sim}")
                compute_heatmap(cfg, simulation_name=condition + '/' + sim,
                                crop=False,
                                if_resize=True,
                                yuv=True,
                                input_model='track1-dave2-uncropped-mc-034',
                                condition=condition,
                                attention_type=attention_type)
