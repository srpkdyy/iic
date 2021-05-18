import numpy as np
import torch
import torchvision as tv
import torchvision.transforms as tf
import torchvision.transforms.functional as tff
from PIL import Image


def my_transforms(x, input_size=224):
    tf_list = [
        tf.ToPILImage(),
        tf.RandomAffine(30, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        tf.RandomPerspective(0.2, 0.9),
        tf.Resize(input_size, input_size),
        tf.RandomHorizontalFlip(),
        tf.RandomVerticalFlip(),
        tf.ColorJitter(0.4, 0.4, 0.4),
        tf.ToTensor()
    ]

    x_tf = x.clone()
    n_batch = x.size(0)
    for i in range(n_batch):
        x_tf[i, 0] = tf.Compose(tf_list)(x_tf[i, 0])

    return x_tf
