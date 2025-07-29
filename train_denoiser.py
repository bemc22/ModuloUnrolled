import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from tqdm import tqdm

from libs.dataloader import Modulo, ModuloDataset, addNoise
from libs.utils import AverageMeter
from models.network_unet import UNetRes as net
from libs.pnp import deep_denoiser

from config import MODEL_SIZE
from config import model_config
# Set device
torch.backends.cudnn.benchmark = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE} device")

# data params
BLUR_KERNEL_SIZE = 1
BLUR_SIGMA = 1

# Constants
datapath = r"data\unmodnet"
datapath_test = r"data\unmodnet_test"
bitdepth = 10
DATA_RANGE = 1.0
STD = (0, 80)
EPOCHS  = 5000
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

# Function to load images
def load_img(path):
    img = np.load(path)
    img = img.astype(np.float32)
    return img

def test(test_loader, model):

    psnr_fn = PeakSignalNoiseRatio(
        data_range=DATA_RANGE, reduction="elementwise_mean", dim=(-3, -2, -1)
    ).to(DEVICE)
    ssim_fn = StructuralSimilarityIndexMeasure(
        data_range=DATA_RANGE, reduction="elementwise_mean"
    ).to(DEVICE)

    psnr_record = AverageMeter()
    ssim_record = AverageMeter()
    loss_record = AverageMeter()

    model.eval()

    with torch.no_grad():
        for (x, y, std), _ in test_loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)
            std = std.to(DEVICE)
            

            y = torch.cat([y, y], dim=1)  # Duplicate the input for the model
            x_hat = deep_denoiser(y, noise_level=std, model=model)

            loss = F.mse_loss(x_hat, x)

            psnr = psnr_fn(x, x_hat).item()
            ssim = ssim_fn(x, x_hat).item()

            psnr_record.update(psnr, x.size(0))
            ssim_record.update(ssim, x.size(0))
            loss_record.update(loss.item(), x.size(0))

    return loss_record, psnr_record, ssim_record

def train_one_epoch(epoch, train_loader, test_loader, model, optimizer, tq=None, scaler=None):

    psnr_fn = PeakSignalNoiseRatio(
        data_range=DATA_RANGE, reduction="elementwise_mean", dim=(-3, -2, -1)
    ).to(DEVICE)
    ssim_fn = StructuralSimilarityIndexMeasure(
        data_range=DATA_RANGE, reduction="elementwise_mean"
    ).to(DEVICE)

    psnr_record = AverageMeter()
    ssim_record = AverageMeter()
    loss_record = AverageMeter()

    model.train()

    for i, batch in enumerate(train_loader):

        (x, y, std), _ = batch
        
        optimizer.zero_grad(set_to_none=True)
        tq.update(1)

        x = x.to(DEVICE)
        y = y.to(DEVICE)
        std = std.to(DEVICE)

        with torch.cuda.amp.autocast(dtype=torch.float16):
            
            y = torch.cat([y, y], dim=1)  # Duplicate the input for the model
            x_hat = deep_denoiser(y, noise_level=std, model=model)
            loss = F.mse_loss(x_hat, x)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            x, x_hat = x.detach(), x_hat.detach()

            psnr = psnr_fn(x, x_hat)
            ssim = ssim_fn(x, x_hat)

            psnr_record.update(psnr, x.size(0))
            ssim_record.update(ssim, x.size(0))
            loss_record.update(loss.item(), x.size(0))

        text = f"Loss: {loss_record}, PSNR: {psnr_record}, SSIM: {ssim_record}"

        tq.set_postfix_str(text)

    test_loss, test_psnr, test_ssim = test(test_loader, model)

    text = f"Tn-Loss:{loss_record}, Tn-PSNR:{psnr_record}, Tn-SSIM:{ssim_record}, Ts-Loss:{test_loss}, Ts-PSNR:{test_psnr}, Ts-SSIM:{test_ssim}"
    tq.set_postfix_str(text)

    save_name = f"denoiser_{MODEL_SIZE}_rgb_no_blur_norm.pth"
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
            # transforms.Grayscale(num_output_channels=n_channels),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            addNoise(std=STD),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Grayscale(num_output_channels=n_channels),
            addNoise(std=STD),
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

    model = net(**model_config).to(DEVICE)
    # preliminar test

    test_loss, test_psnr, test_ssim = test(test_loader, model)
    print(f"Test Loss: {test_loss}, Test PSNR: {test_psnr}, Test SSIM: {test_ssim}")

    epochs = EPOCHS
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-7
    )
    scaler = torch.cuda.amp.GradScaler()

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
                model,
                optimizer,
                tq=tq,
                scaler=scaler,
            )


if __name__ == "__main__":
    main()
