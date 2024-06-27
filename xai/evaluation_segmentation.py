from models.model2 import U_Net
import torch
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
import torchvision.transforms.functional as TF
from torchvision import transforms
import numpy as np
from ThirdEye.ase22.utils import resize
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mapping = {
    150: 0,
    76: 1
}
mappingrgb = {
    76: (1, 0, 255),
    150: (1, 255, 0)
}


def pixelwise_accuracy(img, seg):
    '''
    compare each pixel of predicted seg map with ground truth and returns the accuracy of the prediction
    '''


    assert img.shape == seg.shape, "Images must have the same shape"

    # Flatten images to 1D arrays
    img_flat = img.ravel()
    seg_flat = seg.ravel()

    # Count pixels that match
    correct_pixels = np.sum(img_flat == seg_flat)

    # Calculate accuracy
    accuracy = correct_pixels / len(img_flat)

    return accuracy


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


def preprocess(img):
    '''
    This function turning the input image into Unet model acceptable form
    '''
    # bei bedarf diese Zeile auskommentieren
    # img = img[:, :, :3]

    image = TF.to_tensor(img)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    image = normalize(image)
    image = image.to('cpu')
    image = image.unsqueeze(0)
    return image


def result(predicted_rgb, seg):
    '''
    Calculate the percentage of correct predicted pixel. Returning the same result then the pixelwise_accuracy() fucntion
    '''
    predicted_rgb = resize(predicted_rgb)
    seg = resize(seg)
    all_pixel = predicted_rgb.shape[0] * predicted_rgb.shape[1]
    correct_prediction = 0
    #iteration thru all pixel of the image
    for y in range(predicted_rgb.shape[0]):
        for x in range(predicted_rgb.shape[1]):
            if np.array_equal(predicted_rgb[y, x], [1, 255, 0]) and np.array_equal(seg[y, x], [0, 1, 0]):
                correct_prediction += 1
            elif np.array_equal(predicted_rgb[y, x], [1, 0, 255]) and np.array_equal(seg[y, x], [0, 0, 1]):
                correct_prediction += 1
    return correct_prediction / all_pixel * 100


def evaluation(model, image_folder_path):
    '''
    Show prediction results as histogram and img
    '''
    files = os.listdir(image_folder_path)
    img_files = [f for f in files if f.startswith('image')]
    a = 0
    accuracy = []
    for img_file in img_files:
        a += 1
        print(a)
        img_path = os.path.join(image_folder_path, img_file)

        img = mpimg.imread(img_path)

        segmentation_name = "segmentation" + img_file[5:-3] + "png"
        segmentation_path = os.path.join(image_folder_path, segmentation_name)
        seg = mpimg.imread(segmentation_path)

        x = preprocess(img)
        with torch.no_grad():
            prediction = model(x)

        predicted_rgb = torch.zeros((3, prediction.size()[2], prediction.size()[3])).to('cpu')
        maxindex = torch.argmax(prediction[0], dim=0).cpu().int()
        predicted_rgb = class_to_rgb(maxindex).to('cpu')
        predicted_rgb = predicted_rgb.squeeze().permute(1, 2, 0).numpy()

        # print(result(predicted_rgb, seg))
        accuracy.append(result(predicted_rgb, seg))

        a = 0
        if a % 10:
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(img)
            axs[0].set_title('Image')
            axs[0].axis('off')

            axs[1].imshow(predicted_rgb)
            axs[1].set_title('prediction')
            axs[1].axis('off')

            plt.savefig('/mnt/c/Unet/example of segmentation')
            plt.show()

        a += 1
        '''


    np.save('/mnt/c/Unet/segmentation_dataset/hard_test_result', accuracy)
    plt.hist(accuracy)
    plt.title("average attention heatmaps")

    plt.savefig('/mnt/c/Unet/segmentation_dataset/hard_test_result.png')
'''


if __name__ == '__main__':
    #average result of function evaluation
    '''
    array_path = '/mnt/c/Unet/segmentation_dataset/hard_test_result.npy'
    array = np.load(array_path)

    # Calculate the average value of the array
    average_value = np.mean(array)

    print(f"The average value of the array is: {average_value}")

    exit()
    '''


    

    model = U_Net(3, 2)
    model.to('cpu')
    checkpoint_path = '/mnt/c/Unet/segModels/SegmentationModel_CrossEntropyLoss38.pth'
    model.load_state_dict(torch.load(checkpoint_path, map_location='cuda'))
    model.eval()


    # For generate example
    path = '/mnt/c/Users/Linfe/Downloads/data-ASE2022/benchmark-ASE2022/mutants/udacity_change_label_mutated0_MP_12.5_4/IMG/2022_04_21_17_00_47_568.jpg'

    image = mpimg.imread(path)

    img = image
    image = preprocess(image)
    with torch.no_grad():
        prediction = model(image)

    predicted_rgb = torch.zeros((3, prediction.size()[2], prediction.size()[3])).to('cpu')
    maxindex = torch.argmax(prediction[0], dim=0).cpu().int()
    predicted_rgb = class_to_rgb(maxindex).to('cpu')
    predicted_rgb = predicted_rgb.squeeze().permute(1, 2, 0).numpy()

    fig, axs = plt.subplots(1, 2)
    plt.tight_layout()
    axs[0].imshow(img)
    axs[0].axis('off')

    axs[1].imshow(predicted_rgb)
    axs[1].axis('off')

    plt.savefig('/mnt/c/Unet/fail side normal.svg', bbox_inches='tight')

    plt.show()



    '''
    #just showing images of prediction and original
    path = 'C:/Users/Linfe/Downloads/data-ASE2022/benchmark-ASE2022/ood/xai-track1-fog-10'

    img_files_path = os.path.join(path, 'IMG')
    img_files = os.listdir(img_files_path)
    for img in img_files:
        img_path = os.path.join(img_files_path, img)
        image = mpimg.imread(img_path)

        sm_path = os.path.join(path, 'heatmaps-smoothgrad', 'IMG', 'htm-smoothgrad-' + img)
        fc_path = os.path.join(path, 'heatmaps-faster-scorecam', 'IMG', 'htm-faster-scorecam-' + img)

        sm = mpimg.imread(sm_path)
        fc = mpimg.imread(fc_path)

        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(image)
        axs[0].set_title('Image')
        axs[0].axis('off')

        axs[1].imshow(sm)
        axs[1].set_title('prediction')
        axs[1].axis('off')

        axs[2].imshow(fc)
        axs[2].set_title('prediction')
        axs[2].axis('off')

        plt.show()

    '''

    '''
    path = '/mnt/c/Unet/segmentation_dataset/test'

    evaluation(model, path)
    '''
