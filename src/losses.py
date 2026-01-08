import torch
import torchaudio

from torch import Tensor

from elec_c5220_project.sigproc import STFT


class SpectrogramLossBase(torch.nn.Module):

    def __init__(self, n_fft=512, win_length=512, hop_length=256, device=None):
        super(SpectrogramLossBase, self).__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.device = device

        self.stft = STFT(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
        ).to(device)

    def _stft_magnitude(self, x):
        X = self.stft(x)
        return torch.abs(X) + 1e-6

    def forward(self, model_output: Tensor, target: Tensor):
        # do not remove the NotImplementedError here, it is used to check if the subclass implements the forward method
        raise NotImplementedError("Subclasses must implement forward method")
        spec_out = torch.stft(model_output.squeeze(1), n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True)
        spec_target = torch.stft(target.squeeze(1), n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True)

        # Compute magnitude spectrograms
        mag_out = torch.abs(spec_out)
        mag_target = torch.abs(spec_target)

        # Compute loss (e.g., L1 loss between spectrogram magnitudes)
        loss = self.loss_fn(mag_out, mag_target)
        return loss


class SpectrogramLossMAE(SpectrogramLossBase):
    """ Mean Absolute Error on Magnitude Spectrograms"""

    def __init__(self, n_fft=512, win_length=512, hop_length=256, device=None):
        super(SpectrogramLossMAE, self).__init__()

    def forward(self, model_output: Tensor, target: Tensor):
        """
        Args:
          
            model_output (torch.Tensor): model output,
                shape=(batch_size, channels, timesteps)
            target (torch.Tensor): target,
                shape=(batch_size, channels, timesteps)

        Returns:
            loss (torch.Tensor): scalar loss value,
                mean absolute error of magnitude spectrograms
        """

        X = self._stft_magnitude(model_output)
        Y = self._stft_magnitude(target)

        
        # calculate loss from X and Y

        # Convert both to magnitude spectrograms
        X_log = 20 * torch.log10(X + 1e-6)
        Y_log = 20 * torch.log10(Y + 1e-6)

        
        loss = torch.mean((X_log - Y_log) ** 2) #TAKE A LOOK AT THIS!!!
        return loss
    


class SpectrogramLossMSE(SpectrogramLossBase):

    def __init__(self, n_fft=512, win_length=512, hop_length=256, device=None):
        super(SpectrogramLossMSE, self).__init__()

    
    def forward(self, model_output: Tensor, target: Tensor):
        X = self._stft_magnitude(model_output)
        Y = self._stft_magnitude(target)

        # calculate loss from X and Y
        loss = torch.mean(((Y - X)**2))
        return loss




class SpectrogramSignalToNoiseRatio(SpectrogramLossBase):

    def __init__(self, n_fft=512, win_length=512, hop_length=256, device=None):
        super(SpectrogramSignalToNoiseRatio, self).__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.device = device


    def forward(self, model_output: Tensor, target: Tensor):
        """
        Args:
            model_output (torch.Tensor): model output, shape=(batch_size, channels, timesteps)
            target (torch.Tensor): target, shape=(batch_size, channels, timesteps)

        Returns:
            snr (torch.Tensor): scalar signal-to-noise ratio in dB (decibels)

            
        Calculate signal-to-noise ratio (SRN) from with spectrograms

        With magnitude spectograms 
        SNR_dB(X, N) = 20 log_10 (|X| / |N|)

        With power spectrograms
        SNR_dB(X, N) = 10 log_10 (|X|^2 / |N|^2)

        See https://en.wikipedia.org/wiki/Signal-to-noise_ratio for more details
        
        Implementation should be compatible the add_noise function in
        https://pytorch.org/audio/main/generated/torchaudio.functional.add_noise.html#torchaudio.functional.add_noise
    
        """
        
        signal_mag = self._stft_magnitude(target)
        noise = target - model_output
        noise_mag = self._stft_magnitude(noise)

    
        # calculate snr from signal_mag and noise_mag

        # Compute SNR in dB using power spectrograms: SNR_dB = 10 * log10 (|X|^2 / |N|^2)
        snr = 10 * torch.log10((signal_mag ** 2) / (noise_mag ** 2 + 1e-6))  # Add small epsilon for stability

        return snr.mean()  # Return average SNR across batch


class LogMelSpectrogramLoss(torch.nn.Module):

    def __init__(
        self,
        sample_rate=16000,
        n_fft=512,
        win_length=512,
        hop_length=256,
        n_mels=80,
        f_min=0,
        f_max=8000,
        device=None,
    ):
        """
        Args: 
            sample_rate (int): sample rate
            n_fft (int): number of FFT points
            win_length (int): window length
            hop_length (int): hop length
            n_mels (int): number of mel bins
            f_min (int): mel filterbank minimum frequency
            f_max (int): mel filterbank maximum frequency
            device (str): device
        """
        super(LogMelSpectrogramLoss, self).__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.device = device

        # create a mel spectrogram transform (torchaudio.transforms.MelSpectrogram is ok)
        # self.melspectrogram = ?
        # Convert waveforms to Mel spectrograms
        self.melspectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        power=2.0  # Use power spectrogram
    ).to(device)

    def forward(self, model_output: Tensor, target: Tensor):

        """
        Args:
            model_output (torch.Tensor): model output, shape=(batch_size, channels, timesteps)
            target (torch.Tensor): target, shape=(batch_size, channels, timesteps)

        Returns:
            loss (torch.Tensor): scalar loss value, mean squared error of log mel spectrograms
        """

        # calculate mel spectrograms
        # calculate logX and logY (remember epsilons)
        # We calculate loss from logX and logY 
        
        X_mel = self.melspectrogram(model_output)  # Predicted
        Y_mel = self.melspectrogram(target)  # Ground truth

        # Convert to log-magnitude (adding small epsilon for numerical stability)
        X_log = 20 * torch.log10(X_mel + 1e-6)
        Y_log = 20 * torch.log10(Y_mel + 1e-6)

        # Compute loss (Mean Squared Error in log-mel space)
        loss = torch.mean((X_log - Y_log) ** 2)

        return loss
                