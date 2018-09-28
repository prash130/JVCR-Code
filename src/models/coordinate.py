## Coordinate Regression

import torch.nn as nn


class coordRegressor(nn.Module):
    def __init__(self, nParts):#, input_size, feat_size, output_size):
        super(coordRegressor, self).__init__()
        self.nParts = nParts
        self.depth = 5
        in_ch = [1,64,128,256,512]
        out_ch = [64,128,256,512,256]
        ec_k = [4,4,4,4,4]
        ec_s = [2,2,2,2,1]
        ec_p = [1,1,1,1,0]
        encoder_ = []

        # Adding the Conv layers to reduce the # of channels with every layer [64 -> 128 -> 256 -> 512 -> 256]
        for i in range(self.depth):
            if i < self.depth-1:
                layer_i = nn.Sequential(
                    nn.Conv3d(in_ch[i], out_ch[i], kernel_size=ec_k[i], stride=ec_s[i], padding=ec_p[i]),
                    nn.BatchNorm3d(out_ch[i]),
                    # nn.ReLU(inplace=True)
                    nn.LeakyReLU(0.2, inplace=True)
                    )
            else:
                layer_i = nn.Sequential(
                    nn.Conv3d(in_ch[i], out_ch[i], kernel_size=ec_k[i], stride=ec_s[i], padding=ec_p[i]),
                    # nn.ReLU(inplace=True)
                    # nn.Sigmoid()
                    )
            encoder_.append(layer_i)
        self.encoder = nn.ModuleList(encoder_)

        self.fc_cord = nn.Linear(out_ch[-1], self.nParts*3)
        
  # Start with a 64X64X64 volume (convolves to) -> 1X1X256 -> Fully Connected layer of (68 * 3 points) where 68 is number of landmark points and 3 is x,y,z dimensions.
    def forward(self, x):
        feat = x
        for i in range(self.depth):
            feat = self.encoder[i](feat)
            
        embedding = feat.view(x.size(0), -1)
        cord = self.fc_cord(embedding)
     
        
        return feat, embedding, cord