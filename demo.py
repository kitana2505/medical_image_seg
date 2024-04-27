import os
import numpy as np
from glob import glob
from functools import partial
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torchvision.transforms as TF
from transformers import SegformerForSemanticSegmentation
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

from preprocessing import id2color
import cv2
from tqdm import tqdm
import ipdb


label2color = {
    "Large bowel": (255, 0, 0),  # Red - Large bowel
    "Small bowel": (0, 255, 0),  # Green - Small bowel
    "Stomach": (0, 0, 255),  # Blue - Stomach

}
@dataclass
class Configs:
    NUM_CLASSES: int = 4  # including background.
    CLASSES: tuple = ("Large bowel", "Small bowel", "Stomach")
    IMAGE_SIZE: tuple[int, int] = (288, 288)  # W, H
    MEAN: tuple = (0.485, 0.456, 0.406)
    STD: tuple = (0.229, 0.224, 0.225)
    MODEL_PATH: str = os.path.join(os.getcwd(), "segformer_trained_weights")


def get_model(*, model_path, num_classes):
    model = SegformerForSemanticSegmentation.from_pretrained(model_path, num_labels=num_classes, ignore_mismatched_sizes=True)
    return model


@torch.inference_mode()
def predict(input_image, model=None, preprocess_fn=None, device="cpu"):
    shape_H_W = input_image.size[::-1]
    input_tensor = preprocess_fn(input_image)
    input_tensor = input_tensor.unsqueeze(0).to(device)

    # Generate predictions
    outputs = model(pixel_values=input_tensor.to(device), return_dict=True)
    predictions = F.interpolate(outputs["logits"], size=shape_H_W, mode="bilinear", align_corners=False)

    preds_argmax = predictions.argmax(dim=1).cpu().squeeze().numpy()

    seg_info = [(preds_argmax == idx, class_name) for idx, class_name in enumerate(Configs.CLASSES, 1)]

    return (input_image, seg_info)

def num_to_rgb(seg_info, color_map=label2color):
    image_shape = seg_info[0][0].shape[:2]
    output = np.zeros(image_shape + (3,))

    for k in range(len(seg_info)):
        output[seg_info[k][0]] = color_map[seg_info[k][1]]

    # return a floating point array in range [0.0, 1.0]
    return np.float32(output) / 255.0


## overlay
def image_overlay(input_image, mask):
    alpha = 1.0  # Transparency for the original image.
    beta = 0.7  # Transparency for the segmentation map.
    gamma = 0.0  # Scalar added to each sum.
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
    final_image = cv2.cvtColor(np.array(input_image).astype(np.float32), cv2.COLOR_RGB2BGR) / 255.0

    final_image = cv2.addWeighted(final_image, alpha, mask, beta, gamma, final_image)
    final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
    final_image = np.clip(final_image, 0.0, 1.0)
    return final_image
    
if __name__ == "__main__":
    DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    model = get_model(model_path=Configs.MODEL_PATH, num_classes=Configs.NUM_CLASSES)

    preprocess = TF.Compose(
        [
            TF.Resize(size=Configs.IMAGE_SIZE[::-1]),
            TF.ToTensor(),
            TF.Normalize(Configs.MEAN, Configs.STD, inplace=True),
        ]
    )

    image_paths = glob("./test_imgs/*.png")
    final_images = list()
    input_images = list()
    masks = list()
    for path in image_paths:
        input_image = Image.open(path).convert("RGB")
        input_images.append(input_image)

        input_image, seg_info = predict(input_image, model=model, preprocess_fn=preprocess, device=DEVICE)

        mask = num_to_rgb(seg_info)
        masks.append(mask)
        final_images.append(image_overlay(input_image, mask))



    fig, axes = plt.subplots(len(final_images), 3, figsize=(10, 10))
    for i, (input_image, mask, final_image) in enumerate(zip(input_images, masks, final_images)):
        axes[i][0].imshow(input_image)
        axes[i][0].axis("off")

        axes[i][1].imshow(mask)
        axes[i][1].axis("off")

        axes[i][2].imshow(final_image)
        axes[i][2].axis("off")

    # Set titles for each column
    axes[0][0].set_title("Input Image")
    axes[0][1].set_title("Segmentation Mask")
    axes[0][2].set_title("Overlay")

    # Add legend
    legend_labels = list(label2color.keys())
    legend_colors = [tuple(val/255 for val in c) for c in label2color.values()]

    legend_lines = [Line2D([0], [0], color=c, lw=4) for c in legend_colors]

    fig.legend(legend_lines, legend_labels, loc='lower center', ncol=len(legend_labels))    
    fig.savefig("output.png")
    plt.show()