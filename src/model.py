import torch

from elec_c5220_project.sigproc import STFT

class ResidualBlock(torch.nn.Module):

    def __init__(self, channels: int, kernel_size:int):
        """
        Args:
            channels (int): Number of input, skip and output channels
            kernel_size (int): Kernel size
            skip_inputs (bool): If True, skip connection input tensor is used
        """
        super(ResidualBlock, self).__init__()
        
        
        # define suitable 2D convolutional layers for the residual block
        # self.conv1 =  torch.nn.Conv2d( ? )
        # self.conv2 =  torch.nn.Conv2d( ? )
        
        # First convolution: preserves spatial dimensions using appropriate padding.
        self.conv1 = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.bn1 = torch.nn.BatchNorm2d(channels)
        
        # Second convolution: also uses padding to preserve dimensions.
        self.conv2 = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.bn2 = torch.nn.BatchNorm2d(channels)
        
        self.relu = torch.nn.ReLU(inplace=True)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:

            x (torch.Tensor): Input tensor,
                shape = (batch_size, channels, H, W)
            
        Returns:
            torch.Tensor: Output tensor,
                shape = (batch_size, channels, H, W)
        """

        identity = x  # Save the input tensor for the skip connection

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Add the skip connection
        out = out + identity
        out = self.relu(out)
        
        return out


class ResidualStack(torch.nn.Module):

    def __init__(self, channels: int,
                 kernel_size: int, num_blocks: int) -> None:

        super().__init__()

        self.layers = torch.nn.ModuleList()

        for i in range(num_blocks):
            self.layers.append(ResidualBlock(
                channels = channels, 
                kernel_size = kernel_size,
                ))


    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor,
                shape = (batch_size, channels, H, W)

        Returns:
            torch.Tensor: Output tensor,
                shape = (batch_size, channels, H, W)
        """

        for block in self.layers:
            x = block(x)
        return x

    

class Unsqueeze(torch.nn.Module):
    def forward(self, x):
        # Input shape: (batch_size, in_channels, timesteps)
        # Output shape: (batch_size, in_channels, 1, timesteps)
        return x.unsqueeze(2)

class Squeeze(torch.nn.Module):
    def __init__(self, dim):
        super(Squeeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        if x.size(self.dim) == 1:
            return x.squeeze(self.dim)
        return x 


class ResNet(torch.nn.Module):
    
    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size, num_blocks):
        super(ResNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_blocks = num_blocks

        
        self.input_layer = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=(1, kernel_size),
                stride=1,
                padding=(0, kernel_size // 2)
            ),
            torch.nn.ReLU(inplace=True)
        )
        
        
        self.residual_stack = ResidualStack(
            channels=hidden_channels,
            kernel_size=kernel_size,
            num_blocks=num_blocks
        )
        
        
        self.output_layer = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=out_channels,
                kernel_size=(1, kernel_size),
                stride=1,
                padding=(0, kernel_size // 2)
            ),
            Squeeze(dim=2)  # Remove the singleton dimension at index 2.
        )   

        # define a suitable input layer
        # self.input_layer = ? 

        # define residual stack body network
        # self.residual_stack = ?

        # define a suitable output layer
        # self.output_layer = ?

    def forward(self, x):
        """
        Args: 
            x (torch.Tensor): Input tensor,
                shape = (batch_size, in_channels, timesteps)

        Returns:
            torch.Tensor: Output tensor,
                shape = (batch_size, out_channels, timesteps)
        """
        x = self.input_layer(x)
        x = self.residual_stack(x)
        x = self.output_layer(x)

        return x
    
class MaskedDenoisingResNet(torch.nn.Module):

    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size, num_blocks, n_fft, win_length, hop_length):
        """
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            hidden_channels (int): Number of hidden channels
            kernel_size (int): Kernel size
            num_blocks (int): Number of residual blocks
            n_fft (int): Number of FFT points
            win_length (int): Window length
            hop_length (int): Hop length
        """
        
        super(MaskedDenoisingResNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_blocks = num_blocks
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        self.resnet = ResNet(
            in_channels=n_fft // 2 + 1,  # 257 channels after STFT
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            num_blocks=num_blocks
        )
        
        self.stft = STFT(n_fft=n_fft, win_length=win_length, hop_length=hop_length)
        
    def forward(self, x):
        """
        Args: 
            x (torch.Tensor): Input tensor,
                shape = (batch_size, in_channels, timesteps)

        Returns:
            torch.Tensor: Output tensor,
                shape = (batch_size, out_channels, timesteps)
        """
  
        timesteps = x.shape[-1]
        
        # get complex stft
        X_complex = self.stft(x) 
        magnitude = torch.abs(X_complex)
        
        # transform to suitable input features
        if magnitude.ndim == 4:  # (B, 1, F, T)
            input_features = magnitude.squeeze(1)  # → (B, F, T)
        elif magnitude.ndim == 3:  # already (B, F, T)
            input_features = magnitude
        else:
            raise ValueError(f"Unexpected magnitude shape: {magnitude.shape}")

    
        input_features = input_features.unsqueeze(2)
        
        # apply mask to complex stft

        mask = self.resnet(input_features) 
        mask = torch.sigmoid(mask)
        mask = mask.unsqueeze(1)
        
        # apply inverse STFT
        # save the number of timesteps for the inverse STFT
        X_denoised = X_complex * mask
        x_denoised = self.stft.inverse(X_denoised, length=timesteps)

        return x_denoised

class UpsampleBlock(torch.nn.Module):

    def __init__(self,  in_channels:int, out_channels:int, upsample_factor:int, kernel_size:int):
        """
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            upsample_factor (int): Upsampling factor
            kernel_size (int): Kernel size

        """
        super(UpsampleBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample_factor = upsample_factor
        self.kernel_size = kernel_size

        padding = kernel_size // 2
        output_padding = upsample_factor - 1 if kernel_size % 2 == 1 else 0
        self.upsample = torch.nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=upsample_factor,
            padding=padding,
            output_padding=output_padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor, 
                shape = (batch_size, in_channels, H, W)

        Returns:
            torch.Tensor: Output tensor, 
                shape = (batch_size, out_channels, H', W')
                where H' = H * upsample_factor
                and W' = W * upsample_factor
        """

        x = self.upsample(x)
        return x



class DownsampleBlock(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int, downsample_factor:int, kernel_size:int):
        """ 
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            downsample_factor (int): Downsample factor
            kernel_size (int): Kernel size

        """
        
        super(DownsampleBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample_factor = downsample_factor
        self.kernel_size = kernel_size

        padding = kernel_size // 2
        self.downsample = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                          stride=downsample_factor, padding=padding)


    def forward(self, x: torch.Tensor):

        """
        Args:
            x (torch.Tensor): Input tensor, 
                shape = (batch_size, in_channels, H, W)

        Returns:
            torch.Tensor: Output tensor, 
                shape = (batch_size, out_channels, H', W')
                where H' = H // downsample_factor
                and W' = W // downsample_factor
        
        """
        x = self.downsample(x)
        return x
