import os
import time

import numpy as np
import torch
import torch.nn as nn
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data_loader import HDRDataset
from model import FHDR
from options import Options
from util import make_required_directories, mu_tonemap, save_hdr_image, save_ldr_image
from vgg import VGGLoss

INPUT_PATH = "./dataset/test/LDR"

# initialise options
opt = Options().parse()

# ======================================
# loading data
# ======================================

dataset = HDRDataset(mode="test", opt=opt, input_path = INPUT_PATH)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

print("Testing samples: ", len(dataset))

# ========================================
# model initialising, loading & gpu configuration
# ========================================

model = FHDR(iteration_count=opt.iter)

str_ids = opt.gpu_ids.split(",")
opt.gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        opt.gpu_ids.append(id)

# set gpu device
if len(opt.gpu_ids) > 0:
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() >= len(opt.gpu_ids)

    torch.cuda.set_device(opt.gpu_ids[0])

    if len(opt.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    model.cuda()

mse_loss = nn.MSELoss()

# loading checkpoint for evaluation
model.load_state_dict(torch.load(opt.ckpt_path))

make_required_directories(mode="test")

avg_psnr = 0
avg_ssim = 0

print("Starting evaluation. Results will be saved in '/test_results' directory")

with torch.no_grad():

    for batch, data in enumerate(tqdm(data_loader, desc="Testing %")):

        input = data["ldr_image"].data.cuda()
        path  = str(data["path"])

        #Parse image name
        img_name = path.split('/')[-1].split('.')[0]

        output = model(input)
        output = output[-1]

        for batch_ind in range(len(output.data)):
            save_hdr_image(
                img_tensor=output,
                batch=batch_ind,
                path="./test_results/{}.hdr".format(
                    img_name
                ),
            )
