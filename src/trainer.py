import torch
import torchaudio #1

from torch.utils.data import Dataset
from itertools import cycle


from elec_c5220_project.sigproc import STFT
from elec_c5220_project.data import AudioDataset
from elec_c5220_project.utils import save_checkpoint, load_checkpoint

from torchaudio.functional import add_noise

from torch.utils.tensorboard import SummaryWriter
import matplotlib
from matplotlib import pyplot as plt


class Trainer():

    def __init__(self, 
                 model: torch.nn.Module, 
                 optimizer: torch.optim.Optimizer, 
                 criterion: torch.nn.Module,
                 speech_dataset: AudioDataset, 
                 noise_dataset: AudioDataset,
                 device=None,
                 print_interval = None, 
                 log_dir = None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion.to(device)
        self.speech_dataset = speech_dataset
        self.noise_dataset = noise_dataset
        self.device = device
        self.print_interval = print_interval

        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir)

        self.total_iter = 0

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            win_length=512,
            hop_length=256,
            n_mels=80,
            center=True,
        ).to(device)

    def train(self, iters, batch_size=1, shuffle=True):
        """
        Args:
            iters (int): Number of iterations to train for.
            batch_size (int): Batch size.
            shuffle (bool): Shuffle the dataset.

        Returns:
            list: List of losses.
        """

        model = self.model.to(self.device)
        criterion = self.criterion.to(self.device) 

        speech_dataloader = torch.utils.data.DataLoader(self.speech_dataset, batch_size=batch_size, shuffle=shuffle)
        noise_dataloader = torch.utils.data.DataLoader(self.noise_dataset, batch_size=batch_size, shuffle=shuffle)

        losses = []
        i = 0
        for speech, noise in zip(cycle(speech_dataloader), cycle(noise_dataloader)):
            # move to device
            speech = speech.to(self.device)
            noise = noise.to(self.device)

            # apply noise
            snr = torch.tensor(10.0).reshape(1, 1)
            
            min_batch_size = min(speech.size(0), noise.size(0))
            speech = speech[:min_batch_size]
            noise = noise[:min_batch_size]
            noisy_speech = add_noise(speech, noise, snr=snr)

            
            # zero gradients
            self.optimizer.zero_grad()

            # apply denoising model
            denoised_speech = model(noisy_speech)

            # calculate loss
            loss = criterion(denoised_speech, speech)

            # backward pass
            loss.backward()
            
            # update model weights
            self.optimizer.step()

            losses.append(loss.item())

            if self.print_interval is not None and i % self.print_interval == 0:
                loss = sum(losses) / len(losses)

                print(f'Iter:  {self.total_iter}, Loss: {loss:.4f}')
                self.writer.add_scalar('Loss/train', loss, self.total_iter)

            i += 1
            self.total_iter += 1 

            if i >= iters:
                break

        return losses


    def validate(self, num_minibatches, total_iters, batch_size=1, shuffle=True):

        model = self.model.to(self.device)
        criterion = self.criterion.to(self.device)

        speech_dataloader = torch.utils.data.DataLoader(self.speech_dataset, batch_size=batch_size, shuffle=False)
        noise_dataloader = torch.utils.data.DataLoader(self.noise_dataset, batch_size=batch_size, shuffle=False)

        losses = []
        i = 0
        with torch.no_grad():
            for speech, noise in zip(cycle(speech_dataloader), cycle(noise_dataloader)):
                # move to device
                speech = speech.to(self.device)
                noise = noise.to(self.device)
                # apply noise
                snr = torch.tensor(10.0).reshape(1, 1)
                
                min_batch_size = min(speech.size(0), noise.size(0))
                speech = speech[:min_batch_size]
                noise = noise[:min_batch_size]
                noisy_speech = add_noise(speech, noise, snr=snr)

                # apply denoising model
                denoised_speech = model(noisy_speech)

                # calculate loss
                loss = criterion(denoised_speech, speech)

                losses.append(loss.item())

                # save figures and audio at first iteration
                if i == 0:
                    # plot noisy speech
                    plt.figure(figsize=(10, 4))
                    plt.subplot(1, 3, 1)
                    noisy_speech_mel = torch.log(self.melspec(noisy_speech[0])+1e-9)
                    plt.imshow(noisy_speech_mel[0].detach().cpu().numpy(), aspect='auto', origin='lower')
                    plt.title('Noisy Speech')
                    plt.subplot(1, 3, 2)
                    denoised_speech_mel = torch.log(self.melspec(denoised_speech[0])+1e-9)
                    plt.imshow(denoised_speech_mel[0].detach().cpu().numpy(), aspect='auto', origin='lower')
                    plt.title('Denoised Speech')
                    plt.subplot(1, 3, 3)
                    speech_mel = torch.log(self.melspec(speech[0])+1e-9)
                    plt.imshow(speech_mel[0].detach().cpu().numpy(), aspect='auto', origin='lower')
                    plt.title('Clean Speech')
                    self.writer.add_figure('Mel Spectrograms', plt.gcf(), total_iters)
                    
                    plt.close()

                    # save audio to tensorboard
                    self.writer.add_audio('Noisy Speech', noisy_speech[0].detach().squeeze(), total_iters, sample_rate=16000)
                    self.writer.add_audio('Denoised Speech', denoised_speech[0].detach().squeeze(), total_iters, sample_rate=16000)
                    self.writer.add_audio('Clean Speech', speech[0].detach().squeeze(), total_iters, sample_rate=16000)

                i += 1

                if i >= num_minibatches:
                    break
            
        return sum(losses) / len(losses)
