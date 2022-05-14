""" Full assembly of the parts to form the complete network """

from torch import nn
from torch import optim
from tqdm.auto import tqdm

from .utils import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x, use_gradcam=False):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        if use_gradcam:
            self.activations = x3.detach()
            x3.register_hook(self.activations_hook)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return self.sigmoid(logits)
        

    def activations_hook(self, grads):
        self.gradients = grads

    
    def train(self, train_dataset, val_dataset, epochs=10):

        optimizer = optim.RMSprop(self.parameters(), lr=1e-5, weight_decay=1e-8, momentum=0.9)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
        grad_scaler = torch.cuda.amp.GradScaler(enabled=False)
        criterion = nn.BCELoss()

        for _ in tqdm(range(epochs)):
            for x, target, _ in train_dataset:

                pred = self(x)

                loss = criterion(pred.flatten(), target.flatten()) 

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                
            scheduler.step()
    