import os
import cv2
import torch
import random
# import torchvision.utils as v_utils
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm


def plot_loss(train_loss, val_loss, save_path):
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss, 'bo', label='Training loss')
    plt.plot(train_loss, 'b', label='Training loss')
    plt.plot(val_loss, 'ro', label='Validation loss')
    plt.plot(val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # plt.show()
    plt.savefig(save_path)
    plt.close()
    return save_path


def denormalize(tensor):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    dtype = tensor.dtype
    mean = torch.tensor(mean).to(dtype).reshape(1, 3, 1, 1)
    std = torch.tensor(std).to(dtype).reshape(1, 3, 1, 1)
    return (tensor * std + mean).to('cuda')


class ResultSaver:
    def __init__(self, result_dir='/mnt/c/Unet/segmentation_dataset/result'):
        self.result_dir = result_dir
        # Ensure the result directory exists
        os.makedirs(self.result_dir, exist_ok=True)

    @staticmethod
    def tensor_to_pil(tensor):
        """
        Convert a PyTorch tensor to a PIL Image.
        """
        return Image.fromarray(tensor.byte().cpu().numpy().astype(np.uint8).transpose(1, 2, 0))

    def save_comparison(self, rgb_image, ground_truth, prediction, epoch, idx_batch):
        """
        Save a comparison image with the original, ground truth, and prediction.
        """
        # Convert tensors to PIL images
        # original_pil = self.tensor_to_pil(rgb_image)
        gt_pil = self.tensor_to_pil(ground_truth)
        pred_pil = self.tensor_to_pil(prediction)

        # Create a matplotlib figure
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(rgb_image)
        axes[0].set_title('RGB Image')
        axes[0].axis('off')

        axes[1].imshow(gt_pil)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')

        axes[2].imshow(pred_pil)
        axes[2].set_title('Prediction')
        axes[2].axis('off')

        # Save the figure
        comparison_path = f"{self.result_dir}/comparison_{epoch}_{idx_batch}.png"
        plt.tight_layout()
        plt.savefig(comparison_path)
        plt.close()

        return comparison_path


class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, loss_func, device,
                 result_dir='/mnt/c/Unet/segmentation_dataset/result', checkpoint_path='./unet.pkl'):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.device = device
        self.result_dir = result_dir
        self.checkpoint_path = checkpoint_path

        # Ensure the result directory exists
        os.makedirs(self.result_dir, exist_ok=True)

    def train_epoch(self):
        self.model.train()
        train_loss = 0.0
        for imagergb, labelmask, labelrgb in (tqdm(self.train_dataloader)):
            x, y_ = imagergb.to(self.device), labelmask.to(self.device)

            self.optimizer.zero_grad()
            y = self.model(x)
            y_ = torch.squeeze(y_)

            loss = self.loss_func(y, y_)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        return train_loss / len(self.train_dataloader)

    def validate_epoch(self):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imagergb, labelmask, labelrgb in self.val_dataloader:
                x, y_ = imagergb.to(self.device), labelmask.to(self.device)
                y = self.model(x)
                y_ = torch.squeeze(y_)
                loss = self.loss_func(y, y_)
                val_loss += loss.item()
        return val_loss / len(self.val_dataloader)

    def save_checkpoint(self):
        torch.save(self.model.state_dict(), "unet.pkl")

    def run(self, epochs):
        best_val_loss = float('inf')
        patience = 10
        result_saver = ResultSaver(self.result_dir)
        train_loss = []
        val_loss = []

        for epoch in range(epochs):
            train_loss.append(self.train_epoch())

            val_loss.append(self.validate_epoch())

            print(f"Epoch {epoch}, Train Loss: {train_loss[epoch]}, Validation Loss: {val_loss[epoch]}")
            # self.save_checkpoint()

            # Select a batch of data from the validation set
            val_iter = iter(self.val_dataloader)
            images, masks, _ = next(val_iter)  # Get a batch
            images, masks = images.to(self.device), masks.to(self.device)
            images2, masks2 = images.to('cpu'), masks.to('cpu')

            # Randomly select an image from this batch
            idx = random.randint(0, images.size(0) - 1)
            image, mask = images[idx:idx + 1], masks[idx:idx + 1]  # Select a single image and mask
            image2, mask2 = images2[idx:idx + 1], masks2[idx:idx + 1]

            self.model.eval()
            with torch.no_grad():
                output = self.model(image)

            # Generate thresholded prediction
            # batch, channel, h, w
            # for each mask that created pick highes rate from it match it into result img with rgb

            pred_rgb = torch.zeros((output.size()[0], 3, output.size()[2], output.size()[3])).to(self.device)
            for idx in range(output.size()[0]):
                maxindex = torch.argmax(output[idx], dim=0).cpu().int()
                pred_rgb[idx] = self.train_dataloader.dataset.class_to_rgb(maxindex).to(self.device)

            # Save the comparison image
            image = image2[0]  # Extract a single image from the batch

            # denormalize img
            image = denormalize(image).cpu()
            image = image.squeeze().permute(1, 2, 0).numpy()

            mask_rgb = self.train_dataloader.dataset.class_to_rgb(mask.squeeze(0)).to(self.device)
            pred_rgb = pred_rgb[0]
            #comparison_path = result_saver.save_comparison(image, mask_rgb.cpu(), pred_rgb.cpu(), epoch, idx)
            #print(f"Saved comparison image at {comparison_path}")

            comparison_path_loss = f"{self.result_dir}/comparison_{epoch}.png"
            comparison_path = plot_loss(train_loss, val_loss, comparison_path_loss)
            print(f"Saved loss comparison image at {comparison_path}")

            # Check for early stopping
            epoch_str = str(epoch)

            torch.save(self.model.state_dict(), 'SegmentationModel_CrossEntropyLoss' + epoch_str + '.pth')
        # torch.save(self.model.state_dict(), 'SegmentationModel.pth')  # Save latest model
        plot_loss(train_loss, val_loss, comparison_path_loss)

if __name__ == '__main__':
    #preprocess the udacity2 dataset
    datadir = '/mnt/c/Unet/new_segmentation_dataset/labels/temp'
    for filename in os.listdir(datadir):
        if filename.endswith(".png") or filename.endswith(".jpg"):  # Assuming images are PNG or JPG
            # Read the image
            image_path = os.path.join(datadir, filename)
            image = cv2.imread(image_path)

            # Replace color [0,0,0] with [0,255,0]
            image[(image == [0, 0, 0]).all(axis=2)] = [0, 255, 0]

            # Save the modified image
            cv2.imwrite(image_path, image)
