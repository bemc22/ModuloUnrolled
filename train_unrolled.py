import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from tqdm import tqdm

from torch import nn

import cv2
from libs.dataloader import Modulo, ModuloDataset
from libs.pnp import Unrolled, deep_denoiser
from libs.utils import AverageMeter
from models.network_unet import UNetRes as net

from config import MODEL_SIZE
from config import model_config

from libs.utils import unlimited

import gc  # <-- Required for manual garbage collection

from torchvision.transforms.functional import adjust_sharpness

# Set device
torch.backends.cudnn.benchmark = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE} device")



# Constants
datapath = r"data\unmodnet"
datapath_test = r"data\unmodnet_test"
bitdepth = 10
DATA_RANGE = 2 ** (bitdepth - 8)
STD = (0, 80)
ITERS = 3
EPOCHS  =   2000
BATCH_SIZE = 64
LEARNING_RATE = 1e-6

GAMMA = 1.01
EPSILON = 1e-5



# Function to load images
def load_img(path):
    img = np.load(path)
    img = img.astype(np.float32)
    return img


def test(test_loader, model):

    psnr_fn = PeakSignalNoiseRatio(
        data_range=1.0, reduction="elementwise_mean", dim=(-3, -2, -1)
    ).to(DEVICE)
    ssim_fn = StructuralSimilarityIndexMeasure(
        data_range=1.0, reduction="elementwise_mean"
    ).to(DEVICE)

    psnr_record = AverageMeter()
    ssim_record = AverageMeter()
    loss_record = AverageMeter()

    model.eval()

    tone_mapping = DifferentiableToneMapping(exposure=1.0, factor=0.2, learnable=False).to(DEVICE)

    plot = True

    with torch.no_grad():
        for (x, y, std), _ in test_loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)
            std = std.to(DEVICE)

            x_hat = model(y, std)

            loss = F.mse_loss(x_hat, x)

            x_hat = x_hat / DATA_RANGE
            x = x / DATA_RANGE

            psnr = psnr_fn(x, x_hat).item()
            ssim = ssim_fn(x, x_hat).item()

            if plot:

                x_save = torch.cat([y, x, x_hat], dim=3)
                x_save = x_save.permute(0, 2, 3, 1).cpu().numpy()[0]
                x_save = np.clip(x_save, 0, 1)
                x_save = (x_save * 255).astype(np.uint8)
                cv2.imwrite(f"out.png", x_save)
                plot = False


            psnr_record.update(psnr, x.size(0))
            ssim_record.update(ssim, x.size(0))
            loss_record.update(loss.item(), x.size(0))

    return loss_record, psnr_record, ssim_record

from torch import nn

class DifferentiableToneMapping(nn.Module):
    def __init__(self, exposure=1.0, factor=0.2, learnable=False):
        super().__init__()
        if learnable:
            self.exposure = nn.Parameter(torch.tensor(exposure))
            self.factor = nn.Parameter(torch.tensor(factor))
        else:
            self.register_buffer('exposure', torch.tensor(exposure))
            self.register_buffer('factor', torch.tensor(factor))
        
    def forward(self, x):
        # Apply exposure adjustment
        x = x * self.exposure
        
        # Calculate luminance (ITU-R BT.709 coefficients)
        luminance = 0.2126 * x[:, 0:1, :, :] + 0.7152 * x[:, 1:2, :, :] + 0.0722 * x[:, 2:3, :, :]
        
        # Apply tone-mapping curve to luminance
        tonemapped_lum = luminance / (luminance + self.factor)
        
        # Calculate scaling factor while avoiding division by zero
        scale = tonemapped_lum / (luminance + 1e-6)
        scale = scale.detach()  # Detach to avoid gradients through the scaling factor
        
        # Apply scale to each color channel
        return x * scale
    

def train_one_epoch(
    epoch, train_loader, test_loader, model, optimizer, tq=None, scaler=None
):

    psnr_fn = PeakSignalNoiseRatio(
        data_range=1.0, reduction="elementwise_mean", dim=(-3, -2, -1)
    ).to(DEVICE)
    ssim_fn = StructuralSimilarityIndexMeasure(
        data_range=1.0, reduction="elementwise_mean"
    ).to(DEVICE)

    psnr_record = AverageMeter()
    ssim_record = AverageMeter()
    loss_record = AverageMeter()

    model.train()

    tone_mapping1 = DifferentiableToneMapping(exposure=1.0, factor=0.2, learnable=False).to(DEVICE)
    tone_mapping2 = DifferentiableToneMapping(exposure=1.0, factor=0.01, learnable=False).to(DEVICE)

    for i, batch in enumerate(train_loader):

        (x, y, std), _ = batch
        
        optimizer.zero_grad(set_to_none=True)
        tq.update(1)

        x = x.to(DEVICE)
        y = y.to(DEVICE)
        std = std.to(DEVICE)

        n_trans = 3 # number of data augmentations
        sat_factor = 0.2 # offset for saturation adjustment

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):

            x_hat = model(y, std)

            loss =  F.mse_loss(x, x_hat)

            # apply equivariant regularization
            # loss_eq = 0
            # for j in range(n_trans):

            #     rand_sat = torch.rand(1).item() * sat_factor * 2 - sat_factor
            #     x_sat = x * (1 + rand_sat)
            #     x_sat = x_sat.detach()  # Detach to avoid gradients through the saturation adjustment

            #     std_broad = std.view(-1, 1, 1, 1).expand_as(x_sat)
            #     noise = torch.randn_like(x_sat) * (std_broad / 255.0)
            #     y_sat = unlimited(x_sat + noise, 1.0)

            #     x_est = model(y_sat, std)
            #     loss_eq = F.mse_loss(x_est, x_hat)
            #     scaler.scale(0.01 * loss_eq * (1/n_trans)).backward(retain_graph=True)
            # apply sharpening regularization
            max_val = x.max(dim=0, keepdim=True)[0]
            x_sharp = adjust_sharpness(x / max_val, sharpness_factor=2.0) * max_val
            x_sharp = x_sharp.detach()

            loss_sharp = F.mse_loss(x_sharp, x_hat)
            scaler.scale(0.1 * loss_sharp).backward(retain_graph=True)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        with torch.no_grad():
            x = x / DATA_RANGE
            x_hat = x_hat / DATA_RANGE

            psnr = psnr_fn(x, x_hat)
            ssim = ssim_fn(x, x_hat)

            psnr_record.update(psnr, x.size(0))
            ssim_record.update(ssim, x.size(0))
            loss_record.update(loss.item(), x.size(0))

        text = f"Loss: {loss_record}, PSNR: {psnr_record}, SSIM: {ssim_record}"

        # Free up memory after each batch
        del x, y, std, x_hat, loss
        torch.cuda.empty_cache()
        gc.collect()


        tq.set_postfix_str(text)

    test_loss, test_psnr, test_ssim = test(test_loader, model)

    text = f"Tn-Loss:{loss_record}, Tn-PSNR:{psnr_record}, Tn-SSIM:{ssim_record}, Ts-Loss:{test_loss}, Ts-PSNR:{test_psnr}, Ts-SSIM:{test_ssim}"
    tq.set_postfix_str(text)


    save_name = f"unrolled_{MODEL_SIZE}_supervised_gray.pth"
    # save model
    torch.save(
        model.state_dict(),
        os.path.join("checkpoints", save_name),
    )

    return loss_record.avg, psnr_record.avg, test_loss.avg, test_psnr.avg


def main():

    n_channels = 3

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=n_channels),
            # transforms.GaussianBlur(BLUR_KERNEL_SIZE, BLUR_SIGMA),  
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            Modulo(bitdepth, std=0, train=True),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=n_channels),
            # transforms.GaussianBlur(BLUR_KERNEL_SIZE, BLUR_SIGMA),  
            Modulo(bitdepth, std=0),
        ]
    )

    dataset = ModuloDataset(
        datapath, loader=load_img, extensions=(".npy",), transform=transform
    )
    test_dataset = ModuloDataset(
        datapath_test, loader=load_img, extensions=(".npy",), transform=test_transform
    )

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    model_config["in_nc"]  = n_channels * 2 + 1
    model_config["out_nc"] = n_channels

    model = net(**model_config)

    # model_name =  f"denoiser_{MODEL_SIZE}"
    # model_pool = "checkpoints"
    # model_path = os.path.join(model_pool, model_name + ".pth")
    
    # # # model=torch.load(model_path) # model.load_state_dict(torch.load(model_path), strict=True)
    # model.load_state_dict(torch.load(model_path), strict=True)


    model_unrolled = Unrolled(
        model, deep_denoiser, max_iters=ITERS, gamma=GAMMA, epsilon=EPSILON
    )

    save_name = f"unrolled_{MODEL_SIZE}_supervised_gray.pth"
    model_unrolled.load_state_dict(
        torch.load(os.path.join("checkpoints", save_name) ) , strict=True
    )

    STD_LOAD = (0, 80)

    # model_name =   f"unrolled_{MODEL_SIZE}_supervised.pth"
    # model_pool = "checkpoints"
    # model_path = os.path.join(model_pool, model_name)
    # model_unrolled.load_state_dict(torch.load(model_path), strict=True)

    # core_denoiser_save = f"denoiser_unrolled_{MODEL_SIZE}_rgb_no_blur.pth"
    # torch.save(model_unrolled.model, os.path.join("checkpoints", "backup", core_denoiser_save))
    
    # model_unrolled.freeze_model()
    model_unrolled = model_unrolled.to(DEVICE)

    # preliminar test

    test_loss, test_psnr, test_ssim = test(test_loader, model_unrolled)
    print(f"Test Loss: {test_loss}, Test PSNR: {test_psnr}, Test SSIM: {test_ssim}")

    epochs = EPOCHS
    optimizer = torch.optim.Adam(
        model_unrolled.parameters(), lr=LEARNING_RATE, weight_decay=1e-5
    )
    scaler = torch.cuda.amp.GradScaler()

    data = []

    print("Training...")
    for epoch in range(epochs):

        with tqdm(
            total=len(train_loader),
            dynamic_ncols=True,
            colour="green",
        ) as tq:

            tq.set_description(f"Train :: Epoch: {epoch + 1}/{epochs}")

            loss_val, psnr_val, test_val, test_psnr_val = train_one_epoch(
                epoch,
                train_loader,
                test_loader,
                model_unrolled,
                optimizer,
                tq=tq,
                scaler=scaler,
            )

            row = [loss_val, psnr_val.item(), test_val, test_psnr_val]
            data.append(row)


if __name__ == "__main__":
    main()
