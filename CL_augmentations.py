# util functions
import torch

import torch
import torch.nn.functional as F

def time_warp_augmentation(ecg_batch, num_control_points=4, warp_strength=0.2, seed=None):
    """
    Applies time warping augmentation to ECG signals by stretching or compressing segments nonlinearly.

    Args:
    - ecg_batch (torch.Tensor): A tensor of shape (batch_size, leads_num, signal_length)
    - num_control_points (int): Number of control points for warping (default is 4)
    - warp_strength (float): Maximum relative shift of time indices (default is 0.2)
    - seed (int, optional): Random seed for reproducibility (default is None)

    Returns:
    - augmented_batch (torch.Tensor): The time-warped ECG batch
    """
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    batch_size, leads, signal_length = ecg_batch.shape

    # Generate control points and their time shifts
    control_points = torch.linspace(0, signal_length - 1, steps=num_control_points)
    warp_offsets = (torch.rand(batch_size, leads, num_control_points) * 2 - 1) * warp_strength * signal_length

    # Apply time distortion by warping control points
    warped_control_points = (control_points.unsqueeze(0).unsqueeze(0) + warp_offsets).clamp(0, signal_length - 1)

    # Create interpolation grid
    orig_time = torch.arange(signal_length).repeat(batch_size, leads, 1)  # (B, L, signal_length)
    warped_time = F.interpolate(warped_control_points, size=signal_length, mode="linear", align_corners=True)

    # Interpolate ECG signals at warped time indices
    augmented_batch = F.grid_sample(
        ecg_batch.unsqueeze(1),  # Add a channel dimension
        warped_time.unsqueeze(-1).unsqueeze(-1),  # Convert to 4D grid
        mode="bilinear", align_corners=True
    ).squeeze(1)  # Remove channel dimension

    return augmented_batch



def permutation_augmentation(ecg_batch, m=5, seed=None):
    """
    Applies permutation augmentation by dividing each ECG signal into `m` segments, shuffling them, 
    and concatenating them back together.

    Args:
    - ecg_batch (torch.Tensor): A tensor of shape (batch_size, leads_num, signal_length)
    - m (int): Number of segments to divide each ECG signal into (default is 4)
    - seed (int, optional): Random seed for reproducibility (default is None)

    Returns:
    - augmented_batch (torch.Tensor): The augmented ECG batch with shuffled segments
    """
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    batch_size, leads, signal_length = ecg_batch.shape
    segment_length = signal_length // m  # Length of each segment

    # Ensure signal_length is divisible by m
    if signal_length % m != 0:
        raise ValueError(f"signal_length ({signal_length}) must be divisible by m ({m}).")

    # Create a copy of the batch to modify
    augmented_batch = ecg_batch.clone()

    for i in range(batch_size):
        for j in range(leads):
            # Split the signal into `m` segments
            segments = torch.split(augmented_batch[i, j], segment_length, dim=-1)
            
            # Shuffle the segments
            permuted_indices = torch.randperm(m)
            shuffled_segments = [segments[idx] for idx in permuted_indices]
            
            # Concatenate shuffled segments back
            augmented_batch[i, j] = torch.cat(shuffled_segments, dim=-1)

    return augmented_batch

def zero_masking_augmentation(ecg_batch, ratio=0.1, seed=42):
    """
    Applies zero-masking augmentation by setting consecutive segments to zero.

    Args:
    - ecg_batch (torch.Tensor): A tensor of shape (batch_size, leads_num, signal_length)
    - ratio (float): The proportion of the signal length to mask (default is 10%)
    - seed (int, optional): Random seed for reproducibility (default is None)

    Returns:
    - augmented_batch (torch.Tensor): The augmented ECG batch with masked segments set to zero
    """

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    # Get the shape of the batch
    batch_size, leads, signal_length = ecg_batch.shape
    mask_length = int(ratio * signal_length)
    
    # Create a copy of the original batch to modify
    augmented_batch = ecg_batch.clone()

    for i in range(batch_size):
        for j in range(leads):
            # Randomly select the starting index for the mask
            start_idx = torch.randint(0, signal_length - mask_length + 1, (1,)).item()
            augmented_batch[i, j, start_idx : start_idx + mask_length] = 0.0

    return augmented_batch



def dropout_augmentation(ecg_batch, zero_percent=0.1, seed=42):
    """
    Randomly zero out 10% of the positions in the ECG signals for each lead in the batch.
    
    Args:
    - ecg_batch (torch.Tensor): A tensor of shape (batch_size, leads_num, signal_length)
    - zero_percent (float): The percentage of positions to set to zero (default is 10%)
    
    Returns:
    - augmented_batch (torch.Tensor): The augmented ECG batch with 10% of positions set to zero

    # Example usage:
        batch_size = 64
        leads = 12
        signal_length = 5000
        # Assuming ecg_batch is a tensor with shape (batch_size, leads, signal_length)
        ecg_batch = torch.randn(batch_size, leads, signal_length)  # Simulating a batch of ECG signals

        augmented_batch = dropout_augmentation(ecg_batch, zero_percent=0.1)

        # Verify the augmented batch shape and check some of the signals
        print(f"Original Batch Shape: {ecg_batch.shape}")
        print(f"Augmented Batch Shape: {augmented_batch.shape}")
    """

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    # Get the shape of the batch
    batch_size, leads, signal_length = ecg_batch.shape
    
    # Calculate how many positions to zero out
    num_zero_positions = int(signal_length * zero_percent)
    
    # Create a copy of the original batch to modify
    augmented_batch = ecg_batch.clone()

    # Iterate over each signal in the batch
    for i in range(batch_size):
        for j in range(leads):
            # Randomly select positions to zero out
            zero_indices = torch.randperm(signal_length)[:num_zero_positions]
            augmented_batch[i, j, zero_indices] = 0.0  # Set those positions to zero

    return augmented_batch


import torch



def gaussian_noise_augmentation(ecg_batch, noise_factor=0.05, seed=42):
    """
    Add Gaussian noise to each signal in the ECG batch based on the signal's magnitude.
    
    Args:
    - ecg_batch (torch.Tensor): A tensor of shape (batch_size, leads_num, signal_length)
    - noise_factor (float): A factor that determines the magnitude of the noise relative to the signal's magnitude.
    - seed (int): Random seed for reproducibility.
    
    Returns:
    - augmented_batch (torch.Tensor): The augmented ECG batch with Gaussian noise added.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Compute standard deviation per lead (shape: (batch_size, leads))
    signal_std = ecg_batch.std(dim=2, keepdim=False)  # Compute std per lead

    # Normalize the noise by the signal's standard deviation and multiply by noise_factor
    noise_std = signal_std * noise_factor  # Shape: (batch_size, leads)

    # Generate Gaussian noise for each signal, ensuring correct broadcasting
    noise = torch.randn_like(ecg_batch) * noise_std.view(ecg_batch.shape[0], ecg_batch.shape[1], 1)

    # Add the noise to the original ECG batch
    augmented_batch = ecg_batch + noise
    
    return augmented_batch



def mix_augmentation(ecg_batch, zero_percent=0.1, noise_factor=0.05, seed=42):
    """
    Randomly apply dropout to half of the signals in the ECG batch and Gaussian noise to the other half.
    
    Args:
    - ecg_batch (torch.Tensor): A tensor of shape (batch_size, leads_num, signal_length)
    - zero_percent (float): The percentage of positions to set to zero for dropout (default is 10%)
    - noise_factor (float): The factor that determines the magnitude of the Gaussian noise (default is 0.05)
    
    Returns:
    - augmented_batch (torch.Tensor): The augmented ECG batch with dropout applied to half and Gaussian noise to the other half

    # Example usage:
        batch_size = 64
        leads = 12
        signal_length = 5000
        ecg_batch = torch.randn(batch_size, leads, signal_length)  # Simulating a batch of ECG signals

        # Apply random half augmentation
        augmented_batch = random_half_augmentation(ecg_batch, zero_percent=0.1, noise_factor=0.05)

        # Verify the augmented batch shape and check some of the signals
        print(f"Original Batch Shape: {ecg_batch.shape}")
        print(f"Augmented Batch Shape: {augmented_batch.shape}")
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    
    # Get the batch size
    batch_size = ecg_batch.shape[0]
    
    # Randomly shuffle indices to assign half for dropout and half for Gaussian noise
    indices = torch.randperm(batch_size)
    half_batch_size = batch_size // 2
    
    # Create a clone of the original batch for the augmented signals
    augmented_batch = ecg_batch.clone()

    # Apply dropout to the first half
    dropout_indices = indices[:half_batch_size]
    augmented_batch[dropout_indices] = dropout_augmentation(ecg_batch[dropout_indices], zero_percent, seed=seed)
    
    # Apply Gaussian noise to the second half
    noise_indices = indices[half_batch_size:]
    augmented_batch[noise_indices] = gaussian_noise_augmentation(ecg_batch[noise_indices], noise_factor, seed=seed)
    
    return augmented_batch
