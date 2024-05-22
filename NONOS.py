import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_dims, out_dims, kernel_size, stride, dilation):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dims, out_dims, kernel_size=kernel_size, 
                               stride=stride, dilation=dilation, padding=kernel_size//2, bias=True)
        self.bn = nn.BatchNorm1d(out_dims)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.activation(x)
        return x
    
class SEBlock(nn.Module):
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dims, out_dims//8, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(out_dims//8, out_dims, kernel_size=1, padding=0)
        self.activation = nn.GELU()

    def forward(self, x):
        x_se = nn.functional.adaptive_avg_pool1d(x, 1)
        x_se = self.conv1(x_se)
        x_se = self.activation(x_se)
        x_se = self.conv2(x_se)
        x_se = self.activation(x_se)
        x_out = torch.add(x, x_se)
        return x_out
    
class REBlock(nn.Module):
    def __init__(self, in_dims, out_dims, kernel_size, dilation):
        super().__init__()
        self.ConvBlock1 = ConvBlock(in_dims, out_dims, kernel_size, 1, dilation)
        self.ConvBlock2 = ConvBlock(out_dims, out_dims, kernel_size, 1, dilation)
        self.SEBlock = SEBlock(out_dims, out_dims)


    def forward(self, x):
        x_re = self.ConvBlock1(x)
        x_re = self.ConvBlock2(x_re)
        x_re = self.SEBlock(x_re)
        x_out = torch.add(x, x_re)
        return x_out
    
class NONOS_UNET(nn.Module):
    def __init__(self, input_dim, inner_dim, kernel_size, depth, num_layers):
        super().__init__()

        self.avg_pool1d = nn.ModuleList([nn.AvgPool1d(input_dim, stride=4 ** i) for i in range(1, num_layers-1)])
        
        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        
        self.down_layers.append(self.down_layer(input_dim, inner_dim, kernel_size, 1, depth))
        for i in range(1, num_layers):
            in_channels = inner_dim if i == 1 else (inner_dim * i + input_dim)
            out_channels = inner_dim * (i + 1)
            self.down_layers.append(self.down_layer(in_channels, out_channels, kernel_size, 4, depth))

        for i in range(num_layers, 1, -1):
            self.up_layers.append(ConvBlock(inner_dim * (i + (i-1)), inner_dim * (i - 1), kernel_size, 1, 1))

        self.upsample = nn.Upsample(scale_factor=4, mode='nearest')

        self.outconv = nn.Conv1d(inner_dim, 1, kernel_size=1, stride=1, padding=0)

    def down_layer(self, input_layer, out_layer, kernel, stride, depth):
        block = []
        block.append(ConvBlock(input_layer, out_layer, kernel, stride, 1))
        for _ in range(depth):
            block.append(REBlock(out_layer, out_layer, kernel, 1))
        return nn.Sequential(*block)

    def forward(self, x):
        pool_xs = [avg_pool(x) for avg_pool in self.avg_pool1d]
        
        outs = []
        for i, down_layer in enumerate(self.down_layers):
            x = down_layer(x) if i <= 1 else down_layer(torch.cat([x, pool_xs[i-2]], 1))
            outs.append(x) if i < len(self.down_layers)-1 else None

        for i, up_layer in enumerate(self.up_layers):
            up = self.upsample(x) if i == 0 else self.upsample(up)
            up = torch.cat([up, outs[-(i + 1)]], 1)
            up = up_layer(up)

        out = self.outconv(up)
        return out
