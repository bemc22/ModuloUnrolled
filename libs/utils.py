import torch
import numpy as np
import cv2

# def unlimited(x,  mx):
#     positive_input = x > 0
#     x = x % mx
#     x[ (x == 0 )*(positive_input) ] = mx
#     return x


def unlimited(x, mx):
    return torch.remainder(x, mx) 


def wrapToMax(x, mx):
    return unlimited(x + mx/2, mx) - mx/2



def channel_norm(x):
    x = x - x.min(dim=(-1), keepdim=True)[0].min(dim=(-2), keepdim=True)[0]
    x = x / x.max(dim=(-1), keepdim=True)[0].max(dim=(-2), keepdim=True)[0]
    # x -= x.min()
    # x /= x.max()
    return x


def custom_tone(img):
    hdr = img
    hdr = hdr.astype("float32")
    hdr = (hdr - np.min(hdr)) / (np.max(hdr) - np.min(hdr))
    grayscale = True
    if hdr.ndim == 3:
        if hdr.shape[2] == 3:
            # RGB image (H, W, 3)
            # hdr = cv2.cvtColor(hdr, cv2.COLOR_RGB2BGR)
            grayscale = False
        elif hdr.shape[2] == 1:
            # grayscale image (H, W, 1)
            hdr = hdr[:, :, 0]
    if grayscale:
        hdr = np.stack([hdr, hdr, hdr], axis=2)

    tmo = cv2.createTonemapReinhard(intensity=0.0, light_adapt=0.0, color_adapt=0.0)
    tonemapped = tmo.process(hdr)
    return tonemapped


@torch.jit.script
def qval(sigmaxy, xm, ym, sigma2x, sigma2y):
    return 4 * sigmaxy * xm * ym / ((sigma2x + sigma2y) * (xm**2 + ym**2))


def qindex(x, y):
    # Args:
    #     x: [B, H, W]
    #     y: [B, H, W]
    # Returns:
    #     q: [1]

    x = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], -1)

    xm = x.mean(-1, keepdim=True)
    ym = y.mean(-1, keepdim=True)

    n = x.shape[-1]

    tmpx2 = torch.square(x - xm)
    tmpy2 = torch.square(y - ym)
    tmpxy = torch.mul(x - xm, y - ym)

    sigma2x = tmpx2.sum() / (n - 1)
    sigma2y = tmpy2.sum() / (n - 1)
    sigmaxy = tmpxy.sum() / (n - 1)

    q = qval(sigmaxy, xm, ym, sigma2x, sigma2y)
    return q.mean()


def Q(x, y):
    ws = 16
    if len(x.shape) == 3:
        return torch.mean(torch.tensor([Q(x[i], y[i]) for i in range(x.shape[0])]))

    x_batch = x.unfold(0, ws, 1).unfold(1, ws, 1).reshape(-1, ws, ws)
    y_batch = y.unfold(0, ws, 1).unfold(1, ws, 1).reshape(-1, ws, ws)

    return qindex(x_batch, y_batch)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return "{:.3f} ({:.3f})".format(self.val, self.avg)
