from matplotlib import pyplot
import argparse
import os
import random
import sys
from pathlib import Path
from torch.utils.data import DataLoader, random_split, Dataset
from tqdm import tqdm
import numpy as np
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
import logging
import shutil
import matplotlib.pyplot as plt
import glob
import torch
from torch import Tensor
from torch import optim
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import cv2
import time
import gc


def delete_folder(folder_path):
    """
    Delete a folder and its contents if it exists.

    Parameters:
    folder_path (str): The path of the folder to delete.
    Returns:
    None
    """
    try:
        shutil.rmtree(folder_path)
        print(f"Folder deleted: {folder_path}")
    except Exception as e:
        print(f"Error deleting folder: {e}")

def copy_files(source_folder, destination_folder, n_images_to_copy):
    """
    Copy a specified number of files from a source folder to a destination folder.

    Parameters:
    - source_folder (str): The path to the source folder containing files to copy.
    - destination_folder (str): The path to the destination folder where files will be copied.
    - n_images_to_copy (int): The number of images to copy from the source folder.

    Returns:
    - copied_count (int): The number of files successfully copied.
    """

    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # Get a list of existing files in the destination folder
    existing_files = set(os.listdir(destination_folder))

    # Get a sorted list of source file names
    source_files = sorted(os.listdir(source_folder))

    # Copy the first n images from the source folder, sorted by name
    copied_count = 0

    for i in range(min(n_images_to_copy, len(source_files))):
        source_file_name = source_files[i]
        destination_file_path = os.path.join(destination_folder, source_file_name)

        # Check if the file already exists in the destination folder
        if source_file_name not in existing_files:
            source_file_path = os.path.join(source_folder, source_file_name)

            # Check if the source file exists before copying it
            if os.path.exists(source_file_path):
                shutil.copyfile(source_file_path, destination_file_path)
                copied_count += 1
                existing_files.add(source_file_name)
            else:
                print(f"Source file does not exist: {source_file_path}")

def copy_next_n_images(source_folder, destination_folder, n_images_to_copy, number_m):
    """
    Copy a specified number of unique files from a source folder to a destination folder
    starting from a specified index.

    Parameters:
    - source_folder (str): The path to the source folder containing files to copy.
    - destination_folder (str): The path to the destination folder where files will be copied.
    - n_images_to_copy (int): The number of images to copy from the source folder.
    - number_m (int): The index in the source folder where copying should start.

    Returns:
    - copied_count (int): The number of files successfully copied.
    """

    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # Get a list of existing files in the destination folder
    existing_files = set(os.listdir(destination_folder))

    # Get a sorted list of source file names
    source_files = sorted(os.listdir(source_folder))

    # Copy the next n unique images from the source folder, starting from index number_m
    copied_count = 0

    for i in range(number_m, min(number_m + n_images_to_copy, len(source_files))):
        source_file_name = source_files[i]
        destination_file_path = os.path.join(destination_folder, source_file_name)

        # Check if the file already exists in the destination folder
        if source_file_name not in existing_files:

            # Check if the source file exists before copying it
            source_file_path = os.path.join(source_folder, source_file_name)
            if os.path.exists(source_file_path):
                shutil.copyfile(source_file_path, destination_file_path)
                copied_count += 1
                existing_files.add(source_file_name)
            else:
                print(f"Source file does not exist: {source_file_path}")

    return copied_count


def move_n_images(image_dir, mask_dir, dice_score_image_ids, save_dir_image, save_dir_mask, n=10):
    """
    Save N images based on Dice score from the experiment to a new directory,
    and delete them from the original directory.

    Parameters:
    - image_dir: Directory containing original images.
    - mask_dir: Directory containing original masks.
    - dice_score_image_ids: List containing tuples of (Dice score, image ID).
    - save_dir_image: Directory to save selected images.
    - save_dir_mask: Directory to save selected masks.
    - n: Number of top images to save (default is 10).
    """

    # Sort dice_score_image_ids based on Dice score in ascending order
    sorted_images = sorted(dice_score_image_ids, key=lambda x: x[0], reverse=False)[:n]

    for score, image_id in sorted_images:
        # Form the file paths for the selected image and mask
        image_path = os.path.join(image_dir, f"{image_id[0]}.jpg")
        mask_path = os.path.join(mask_dir, f"{image_id[0]}_Segmentation.png")

        os.makedirs(save_dir_image, exist_ok=True)
        os.makedirs(save_dir_mask, exist_ok=True)

        # Copy the selected image and mask to the new directory
        shutil.copy(image_path, os.path.join(save_dir_image, f"{image_id[0]}.jpg"))
        shutil.copy(mask_path, os.path.join(save_dir_mask, f"{image_id[0]}_Segmentation.png"))

        # Delete the selected image and mask from the original directory
        os.remove(image_path)
        os.remove(mask_path)


def load_image(filename):
    """
    Load an image from a file.

    Parameters:
    - filename (str): The path to the image file.

    Returns:
    - Image: A PIL Image object.
    """

    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    """
    Get unique mask values from a mask file.

    Parameters:
    - idx (str): The index or name of the data example.
    - mask_dir (Path): The directory containing mask files.
    - mask_suffix (str): The suffix for mask filenames.

    Returns:
    - numpy.ndarray: An array of unique mask values.
    """

    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        """
        Create a basic dataset for image segmentation.

        Parameters:
        - images_dir (str): The directory containing input images.
        - mask_dir (str): The directory containing mask images.
        - scale (float): The scale factor to resize images and masks.
        - mask_suffix (str): The suffix for mask filenames.

        Returns: None
        """

        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        print(f'Creating dataset with {len(self.ids)} examples')
        print('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        print(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        """
        Preprocess images and masks by resizing and normalizing.

        Parameters:
        - mask_values (list): List of unique mask values.
        - pil_img (PIL.Image): The input image or mask.
        - scale (float): The scale factor for resizing.
        - is_mask (bool): True if the input is a mask, False for an image.

        Returns:
        - numpy.ndarray: The preprocessed image or mask.
        """

        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous(),
            'image_id': name
        }


def collate_fn(batch):
    images = [item['image'] for item in batch]
    masks = [item['mask'] for item in batch]
    image_ids = [item['image_id'] for item in batch] 
    # Find maximum height and width in the batch
    max_height = min(image.shape[1] for image in images)
    max_width = min(image.shape[2] for image in images)

    # Pad images and masks to the maximum height and width
    images_padded = [F.pad(image, (0, max_width - image.shape[2], 0, max_height - image.shape[1])) for image in images]
    masks_padded = [F.pad(mask, (0, max_width - mask.shape[1], 0, max_height - mask.shape[0])) for mask in masks]

    # Convert padded images and masks to a tensor
    images_tensor = torch.stack(images_padded)
    masks_tensor = torch.stack(masks_padded)

    return {'image': images_tensor, 'mask': masks_tensor, 'image_id': image_ids}


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        """
        Create a dataset specific to the Carvana image segmentation task.

        Parameters:
        - images_dir (str): The directory containing input images.
        - mask_dir (str): The directory containing mask images.
        - scale (float): The scale factor to resize images and masks.

        Returns: None
        """

        super().__init__(images_dir, mask_dir, scale, mask_suffix='_Segmentation')

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    """
    Calculate the Dice coefficient.

    Parameters:
    - input (Tensor): Predicted binary mask.
    - target (Tensor): Ground truth binary mask.
    - reduce_batch_first (bool): If True, compute the Dice coefficient for each batch and then average.
    - epsilon (float): Smoothing factor to prevent division by zero.

    Returns:
    - Tensor: The Dice coefficient.
    """

    # Check input and target shapes
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    # Define dimensions to sum over
    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    # Calculate intersection and union
    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    # Calculate Dice coefficient
    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    """
    Calculate the Dice coefficient for multiclass segmentation.

    Parameters:
    - input (Tensor): Predicted multiclass mask.
    - target (Tensor): Ground truth multiclass mask.
    - reduce_batch_first (bool): If True, compute the Dice coefficient for each batch and then average.
    - epsilon (float): Smoothing factor to prevent division by zero.

    Returns:
    - Tensor: The Dice coefficient for multiclass segmentation.
    """

    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)

def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    """
    Calculate the Dice loss.

    Parameters:
    - input (Tensor): Predicted mask.
    - target (Tensor): Ground truth mask.
    - multiclass (bool): If True, compute multiclass Dice loss.

    Returns:
    - Tensor: The Dice loss.
    """

    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

@torch.inference_mode()
def evaluate(net, dataloader, device, amp=True):
    """
    Evaluate a neural network model on a validation dataset and collect analysis data.

    Parameters:
    - net: The neural network model to evaluate.
    - dataloader: The data loader for the validation dataset.
    - device: The device (e.g., CPU or GPU) to run inference on.
    - amp: A flag indicating whether to use automatic mixed precision for faster training (if supported).

    Returns:
    - average_dice_score: The average Dice score across all batches.
    - dice_score_image_ids: A list containing tuples of (Dice score, image ID) for each batch.
    """

    net.eval()
    num_val_batches = len(dataloader)
    dice_score_image_ids = []

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true, image_id = batch['image'], batch['mask'], batch['image_id']

            image = image.to(device=device, dtype=torch.float32)
            mask_true = mask_true.to(device=device, dtype=torch.long)
            mask_pred = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask values should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                dice = dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                dice_score_image_ids.append((dice.item(), image_id))
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes]'
                mask_true = F.one_hot(mask_true.to(torch.int64), net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                dice = multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
                dice_score_image_ids.append((dice.item(), image_id))  # Append the Dice score as a float

    net.train()

    # Extract Dice scores from the list
    dice_scores = [score for score, _ in dice_score_image_ids]

    average_dice_score = sum(dice_scores) / max(num_val_batches, 1)  # Calculate the average Dice score
    return average_dice_score, dice_score_image_ids

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    """
    Make predictions on a single input image using a trained neural network model.

    Parameters:
    - net: The trained neural network model.
    - full_img: The input image to make predictions on.
    - device: The device (e.g., CPU or GPU) to run predictions on.
    - scale_factor: Scaling factor for the input image.
    - out_threshold: Threshold for the output mask.

    Returns:
    - mask: The predicted mask as a NumPy array.
    """

    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()

def copy_corresponding_random_images_and_delete(folder1, folder2, folder3, folder4, n_images_to_copy):
    os.makedirs(folder3, exist_ok=True)
    os.makedirs(folder4, exist_ok=True)

    # Get a list of files in folder1
    files1 = os.listdir(folder1)
    random.shuffle(files1)
    files_to_copy1 = random.sample(files1, n_images_to_copy)

    # Iterate through randomly selected files in folder1
    for filename in files_to_copy1:
        # Form the corresponding file paths in folder2
        file2 = os.path.join(folder2, filename.replace('.jpg', '_Segmentation.png'))

        # Check if the corresponding file in folder2 exists
        if os.path.exists(file2):
            # Form the file paths in folder3 and folder4
            file1_path = os.path.join(folder1, filename)
            file3_path = os.path.join(folder3, filename)
            file4_path = os.path.join(folder4, filename.replace('.jpg', '_Segmentation.png'))

            # Copy files from folder1 and folder2 to folder3 and folder4
            shutil.copy(file1_path, file3_path)
            shutil.copy(file2, file4_path)

            # Delete the files from folder1 and folder2
            os.remove(file1_path)
            os.remove(file2)

    return folder3, folder4

def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)

def load_image(file_path):
    return Image.open(file_path)

def save_image(image, output_path):
    # Ensure the output directory exists, create it if not
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)

def create_directory(directory):
    # Ensure the directory exists, create it if not
    os.makedirs(directory, exist_ok=True)

def predict_and_save(net, input_path, output_path, scale_factor=0.5, out_threshold=0.5, device=None):
    # Load the image
    img = load_image(input_path)

    # Predict mask
    mask = predict_img(net=net, full_img=img, scale_factor=scale_factor, out_threshold=out_threshold, device=device)

    # Convert mask to image
    result = mask_to_image(mask, [0, 1])

    # Save the result
    save_image(result, output_path)

def composite_images(original_path, pred_path, output_path):
    im1 = load_image(original_path)
    im2 = load_image(pred_path)
    mask = load_image(pred_path)

    # Ensure the images have the same size
    if im1.size != im2.size:
        raise ValueError("Image sizes do not match.")

    # Composite images
    composed_img = Image.composite(im1.convert('RGB'), im2.convert('RGB'), mask)

    # Save the composed image with directory creation
    save_image(composed_img, output_path)

def train_model(
        name,
        dir_checkpoint,
        train_image,
        train_mask,
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    """
    Train a neural network model on a dataset with validation round at the end of each epoch and checkpointing.

    Parameters:
    - model: The neural network model to train.
    - device: The device (e.g., CPU or GPU) to run training on.
    - epochs: Number of training epochs.
    - batch_size: Batch size for training.
    - learning_rate: Learning rate for optimization.
    - val_percent: Percentage of data to use for validation.
    - save_checkpoint: Whether to save model checkpoints during training.
    - img_scale: Scaling factor for input images.
    - amp: Whether to use automatic mixed precision for faster training.
    - weight_decay: Weight decay for optimization.
    - momentum: Momentum for optimization.
    - gradient_clipping: Gradient clipping threshold.

    Returns:
    - None
    """
    # 1. Create dataset
    try:
        dataset = CarvanaDataset(train_image, train_mask, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(train_image, train_mask, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=1, pin_memory=True, collate_fn=collate_fn)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    print(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks, image_id = batch['image'], batch['mask'], batch['image_id']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                epoch_loss += loss.item()

        # Calculate average training loss for the epoch
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f'Training Loss: {avg_epoch_loss}')

        # Get the current learning rate from the optimizer
        current_lr = optimizer.param_groups[0]['lr']

        # Validation round at the end of the epoch
        average_dice_score, dice_score_image_ids = evaluate(model, val_loader, device, amp)
        scheduler.step(average_dice_score)

        print('Validation DICE loss: {} for epoch {}'.format(average_dice_score, epoch))
        print('Learning rate: {}'.format(current_lr))

        # Paths and directory
        input_image_path = "/home/it21918/data/evaluate_image_100/ISIC_0011150.jpg"
        dir_checkpoint = f"/home/it21918/data/checkpoints/{name}/"
        output_pred_path = f"{dir_checkpoint}ISIC_0011150_pred.jpg"
        output_composed_path = f"{dir_checkpoint}ISIC_0011150_composed.jpg"

        # Predict and save
        predict_and_save(model, input_image_path, output_pred_path, device=device)

        # Composite images
        composite_images(input_image_path, output_pred_path, output_composed_path)

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, dir_checkpoint + 'checkpoint_epoch' + str(epoch) + '.pth')
            print(f'Checkpoint {epoch} saved!')


def train_evaluate_with_worse_exp_images(
    name,
    model,
    exp_image_dir,
    exp_mask_dir,
    train_image_base,
    train_mask_base,
    eval_image_base,
    eval_mask_base,
    num_exp_images,
    num_train_images,
    step,
    num_epochs,
    batch_size,
    learning_rate,
    device,
    val_percent,
    img_scale,
    amp,
    weight_decay,
):
    for i in range(step, num_exp_images, step):
        try:
            dataset = CarvanaDataset(exp_image_dir, exp_mask_dir, img_scale)
        except (AssertionError, RuntimeError):
            dataset = BasicDataset(exp_image_dir, exp_mask_dir, img_scale)

        loader_args = dict(batch_size=1, num_workers=2, pin_memory=True)
        val_loader = DataLoader(dataset, shuffle=False, drop_last=True, **loader_args)

        average_dice_score, dice_score_image_ids = evaluate(model, val_loader, device, amp)

        train_image_dir = f"/home/it21918/data/{name}/selected/train_{num_train_images}_exp_{i}_image"
        train_mask_dir = f"/home/it21918/data/{name}/selected/train_{num_train_images}_exp_{i}_mask"
        
        copy_files(train_image_base, train_image_dir, num_train_images)
        copy_files(train_mask_base, train_mask_dir, num_train_images)

        move_n_images(exp_image_dir, exp_mask_dir, dice_score_image_ids, train_image_dir, train_mask_dir, i)

        while not (
            len(os.listdir(train_image_dir)) == (i + num_train_images) or
            len(os.listdir(train_mask_dir)) == (i +  num_train_images)
        ):
            time.sleep(1)

        # Train the model with the new images and masks
        train_model(
            name=f"{name}/selected/train_{num_train_images}_exp_{i}/",
            dir_checkpoint=f"/home/it21918/data/checkpoints/{name}/selected/train_{num_train_images}_exp_{i + step}/",
            train_image=train_image_dir,
            train_mask=train_mask_dir,
            model=model,
            epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            val_percent=val_percent,
            save_checkpoint=True,
            img_scale=img_scale,
            amp=amp,
            weight_decay=weight_decay
        )

        # Evaluate the model with the new images and masks
        try:
            dataset = CarvanaDataset(eval_image_base, eval_mask_base, img_scale)
        except (AssertionError, RuntimeError):
            dataset = BasicDataset(eval_image_base, eval_mask_base, img_scale)

        loader_args = dict(batch_size=1, num_workers=2, pin_memory=True)
        val_loader = DataLoader(dataset, shuffle=False, drop_last=True, **loader_args)

        average_dice_score, dice_score_image_ids = evaluate(model, val_loader, device, amp)
        print(f"Average Dice Score for exp_{i}:", average_dice_score)

# Check if a CUDA-compatible GPU is available, and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
torch.cuda.empty_cache()
gc.collect()

# Load the model architecture
model = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=False, scale=0.5)
model.to(device=device);

checkpoint = '/home/it21918/data/checkpoints/exp2/train_100/checkpoint_epoch10.pth' 
state_dict = torch.load(checkpoint, map_location=device)
mask_values = state_dict.pop('mask_values', [0.1])
model.load_state_dict(state_dict);
model.to(device=device);

exp_img_dir = "/home/it21918/data/exp3_image_selected_1079"
exp_mask_dir = "/home/it21918/data/exp3_mask_selected_1079"


train_evaluate_with_worse_exp_images(
    name= "exp3_2/100",
    model=model,
    exp_image_dir=exp_img_dir,
    exp_mask_dir=exp_mask_dir,
    train_image_base="/home/it21918/data/train_image_100",
    train_mask_base="/home/it21918/data/train_mask_100",
    eval_image_base="/home/it21918/data/evaluate_image_100",
    eval_mask_base="/home/it21918/data/evaluate_mask_100",
    num_exp_images=301,
    num_train_images=100,
    step=50,
    num_epochs=10,
    batch_size=8,
    learning_rate=1e-5,
    device=device,
    val_percent=0.2,
    img_scale=0.25,
    amp=True,
    weight_decay=1e-8,
)
