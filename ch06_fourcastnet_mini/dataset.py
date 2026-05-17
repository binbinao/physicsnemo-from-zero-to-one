"""
Chapter 6: Weather Window Dataset
====================================
PyTorch Dataset that creates (input, target) pairs from weather time series.

Each sample: input = state at time t, target = state at time t+1.
Supports configurable lead time and windowing.
"""

import os
import sys
import torch
from torch.utils.data import Dataset


class WeatherWindowDataset(Dataset):
    """
    Dataset for next-step weather prediction.
    
    Given a tensor of shape (T, C, H, W), creates pairs:
        input  = data[t]     → shape (C, H, W)
        target = data[t + lead_time]  → shape (C, H, W)
    
    Args:
        data: tensor of shape (T, C, H, W)
        lead_time: number of steps ahead to predict (default: 1)
    """

    def __init__(self, data: torch.Tensor, lead_time: int = 1):
        self.data = data
        self.lead_time = lead_time
        self.n_samples = len(data) - lead_time

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx], self.data[idx + self.lead_time]


def load_weather_dataset(data_dir: str = "data", lead_time: int = 1,
                          train_frac: float = 0.8):
    """
    Load toy weather data and create train/test datasets.
    
    Returns:
        train_dataset, test_dataset
    """
    path = os.path.join(data_dir, "toy_weather.pt")
    if not os.path.exists(path):
        # Generate on the fly
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
        from generate_toy_weather import generate_toy_weather
        print("Generating toy weather data...")
        result = generate_toy_weather()
        os.makedirs(data_dir, exist_ok=True)
        torch.save(result, path)
        data = result['data']
    else:
        result = torch.load(path, weights_only=False)
        data = result['data']

    n_train = int(train_frac * len(data))
    train_ds = WeatherWindowDataset(data[:n_train], lead_time)
    test_ds = WeatherWindowDataset(data[n_train:], lead_time)

    print(f"Weather dataset: train={len(train_ds)}, test={len(test_ds)}, "
          f"shape=({data.shape[1]}, {data.shape[2]}, {data.shape[3]})")
    return train_ds, test_ds


if __name__ == "__main__":
    train_ds, test_ds = load_weather_dataset()
    x, y = train_ds[0]
    print(f"Sample input:  {x.shape}")
    print(f"Sample target: {y.shape}")
