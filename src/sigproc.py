import torch
import torchaudio

class STFT(torch.nn.Module):

    def __init__(self, n_fft, win_length, hop_length, center=True):
        """
        Args:
            n_fft (int): Number of Fourier bins
            win_length (int): Window length
            hop_length (int): Hop length
            center (bool): If True, the time-domain signal is padded so that frame t is centered at time t * hop_length
        """

        super(STFT, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.hann_window(win_length)
        self.center = center

    def forward(self, x):
        """ 
        Args:
            x (torch.Tensor): Input tensor, shape = (batch, channels, time)
        Returns:
            torch.Tensor: Complex-valued STFT, shape = (batch, channels, n_fft, frames)
        """

        batch, channels, time = x.shape
        if channels > 1:
            raise ValueError("Input tensor must have only one channel")
        
        X = torch.stft(
            x.squeeze(1),
            n_fft = self.n_fft,
            hop_length = self.hop_length,
            win_length = self.win_length,
            window = self.window.to(x.device),
            center = self.center,
            return_complex = True
        )
        
        return X.unsqueeze(1)


    def inverse(self, X, length=None):
        """
        Args:
            X (torch.Tensor): Complex-valued STFT, shape = (batch, channels, n_fft, frames)
            length (int): Length of the output signal
        Returns:
            torch.Tensor: Inverse STFT, shape = (batch, channels, time)
        """

        batch, channels, fbins, frames = X.shape
        if channels > 1:
            raise ValueError("Input tensor must have only one channel")
        
        X = X.squeeze(1)
        
        x = torch.istft(
            X,
            n_fft = self.n_fft,
            hop_length = self.hop_length,
            win_length = self.win_length,
            window = self.window.to(X.device),
            center = self.center,
            length = length
        )
        
        return x.unsqueeze(1)