import os

import librosa
from torch.utils import data

from util.utils import sample_fixed_length_data_aligned
import random
import tqdm

class Dataset(data.Dataset):
    def __init__(self,
                 dataset,
                 limit=None,
                 offset=0,
                 sample_length=16384,
                 mode="train"):
        """Construct dataset for training and validation.
        Args:
            dataset (str): *.txt, the path of the dataset list file. See "Notes."
            limit (int): Return at most limit files in the list. If None, all files are returned.
            offset (int): Return files starting at an offset within the list. Use negative values to offset from the end of the list.
            sample_length(int): The model only supports fixed-length input. Use sample_length to specify the feature size of the input.
            mode(str): If mode is "train", return fixed-length signals. If mode is "validation", return original-length signals.

        Notes:
            dataset list file：
            <noisy_1_path><space><clean_1_path>
            <noisy_2_path><space><clean_2_path>
            ...
            <noisy_n_path><space><clean_n_path>

            e.g.
            /train/noisy/a.wav /train/clean/a.wav
            /train/noisy/b.wav /train/clean/b.wav
            ...

        Return:
            (mixture signals, clean signals, filename)
        """
        super(Dataset, self).__init__()
        dataset_list = [line.rstrip('\n') for line in open(os.path.abspath(os.path.expanduser(dataset)), "r")]

        dataset_list = dataset_list[offset:]
        if limit:
            dataset_list = dataset_list[:limit]

        assert mode in ("train", "validation"), "Mode must be one of 'train' or 'validation'."

        self.dataset_list = dataset_list
        self.sample_length = sample_length
        self.mode = mode

        segments = []
        for dataset in tqdm.tqdm(dataset_list):
            filename = os.path.splitext(os.path.basename(dataset))[0]
            full_audio, sr = librosa.load(dataset, sr=None)
            assert sr == 16000, "sr is %d, should be 16000" % sr
            full_audio = (full_audio - full_audio.min()) / (full_audio.max() - full_audio.min()) # minmax
            start = 0
            while start + sample_length <= len(full_audio):
                segments.append((full_audio[start: start + sample_length], 
                                "%s_%d".format(filename, start//sample_length)))
                start += sample_length

        random.seed(0)
        random.shuffle(segments)
        self.segments = segments
        self.length = len(segments)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        segment, name = self.segments[item]
        return segment.reshape(1, -1), segment.reshape(1, -1), name