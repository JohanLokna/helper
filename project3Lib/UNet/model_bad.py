""" Full assembly of the parts to form the complete network """

from torch import nn

from .utils import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, depth = 4, cf = 64, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1


        self.inc = DoubleConv(n_channels, cf)

        # Add down sampling
        self.downs = []
        for _ in range(depth - 1):
            self.downs.append(Down(cf, 2 * cf))
            cf *= 2
        self.downs.append(Down(cf, 2 * cf // factor))
        cf *= 2

        # Add upsampling
        self.ups = []
        for _ in range(depth - 1):
            self.ups.append(Up(cf, cf // (2 * factor), bilinear))
            cf = cf // 2
        self.ups.append(Up(cf, cf // 2, bilinear))
        cf = cf // 2
        
        # Add outputs
        self.outc = OutConv(cf, n_classes)
        self.sigmoid = nn.Sigmoid()

        # self.down1 = Down(64, 128)
        # self.down2 = Down(128, 256)
        # self.down3 = Down(256, 512)
        # self.down4 = Down(512, 1024 // factor)
        # self.up1 = Up(1024, 512 // factor, bilinear)
        # self.up2 = Up(512, 256 // factor, bilinear)
        # self.up3 = Up(256, 128 // factor, bilinear)
        # self.up4 = Up(128, 64, bilinear)
        # self.outc = OutConv(64, n_classes)
        # self.sigmoid = nn.Sigmoid()
    
    def to(self, device):
        for down in self.downs:
            down.to(device=device)
        for up in self.ups:
            up.to(device=device)
        super().to(device=device)


    def forward(self, x, use_gradcam=False):
        # x1 = self.inc(x)
        # x2 = self.down1(x1)
        # x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x5 = self.down4(x4)

        # if use_gradcam:
        #     self.activations = x3.detach()
        #     x3.register_hook(self.activations_hook)

        # x = self.up1(x5, x4)
        # x = self.up2(x, x3)
        # x = self.up3(x, x2)
        # x = self.up4(x, x1)
        # logits = self.outc(x)
        # return self.sigmoid(logits)

        states = [self.inc(x)]
        for down in self.downs:
            states.append(down(states[-1]))

        x = self.ups[0](states[-1], states[-2])
        for i, up in enumerate(self.ups[1:]):
            x = up(x, states[-(3 + i)])
        
        logits = self.outc(x)
        return self.sigmoid(logits)
        

    def activations_hook(self, grads):
        self.gradients = grads