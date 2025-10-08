import torch.nn as nn

class SNet(nn.Module):
    def __init__(self,n_chan,chan_embed=64):
        super(SNet, self).__init__()

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv3d(n_chan,chan_embed,3, padding=1)
        self.conv2 = nn.Conv3d(chan_embed, chan_embed, 3, padding=1)
        self.conv3 = nn.Conv3d(chan_embed, n_chan, 1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)

        return x