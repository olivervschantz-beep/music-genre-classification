import torch
import glob
import os
import torchaudio
import torch.nn.functional as F  # Added import for F.pad

class AudioDataset(torch.utils.data.Dataset):

    def __init__(self, wav_dir, segment_len, sample_rate, file_ext="*.wav", file_list=None):
        self.wav_dir = wav_dir
        self.segment_len = segment_len
        self.sample_rate = sample_rate
        self.file_ext = file_ext
        self.file_list = file_list

        self.wav_files = self._get_wav_files()

    def _get_wav_files(self):
        if self.file_list is not None:
            wav_files = [os.path.join(self.wav_dir, f) for f in self.file_list]
        else:
            wav_files = glob.glob(os.path.join(self.wav_dir, self.file_ext))
        if len(wav_files) == 0:
            raise ValueError(f"No wav files found in {self.wav_dir} with extension {self.file_ext}")
        return sorted(wav_files)

    def __getitem__(self, index):
        """ 
        Args:
            index (int): Index
        Returns:
            torch.Tensor: Waveform, shape = (1, segment_len)
        """
        waveform, sample_rate = torchaudio.load(self.wav_files[index])
        if sample_rate != self.sample_rate:
            raise ValueError(f"Sample rate of waveform {index} is {sample_rate}. Expected {self.sample_rate}.")
        
        if self.segment_len is None:
            return waveform

        # Ensure waveform is mono: if multiple channels, average them.
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        num_samples = waveform.shape[1]

        # random crop if longer than segment_len
        if num_samples > self.segment_len:
            max_offset = num_samples - self.segment_len
            start = torch.randint(0, max_offset + 1, (1,)).item()
            waveform = waveform[:, start:start + self.segment_len]
        # pad if shorter than segment_len
        elif num_samples < self.segment_len:
            pad_length = self.segment_len - num_samples
            waveform = F.pad(waveform, (0, pad_length))

        return waveform

    def __len__(self):
        return len(self.wav_files)
    