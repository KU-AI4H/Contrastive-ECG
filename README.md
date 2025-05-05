# Self-Supervised Pretraining of ECG Signals Using Contrastive Learning

## Pretraining Phase

- Self-supervised pretraining of ECG signals using contrastive learning.
- Pretrain on a massive unlabeled dataset.

## Contrastive Augmentation Methods 

- **Time Wrapping Augmentation**: Stretches and compresses alternating segments of ECG signals.
- **Permutation Augmentation**: Divides each ECG signal into `m` segments, shuffles them, and concatenates them back together.
- **Zero Masking Augmentation**: Sets consecutive segments of the ECG signal to zero.
- **Dropout Augmentation**: Randomly zeros out 10% of the positions in each lead of the ECG signals within a batch.
- **Gaussian Noise Augmentation**: Adds Gaussian noise to each signal based on its magnitude.
- **CLOCKS**: Based on Kiyasseh et al., *"Clocs: Contrastive learning of cardiac signals across space, time, and patients,"* International Conference on Machine Learning (ICML), PMLR, 2021.
- Find them in `CL_augmentations.py`

## Encoder Architectures 

- Implements various encoder architectures specifically designed for ECG time series (`models.py`):
  - **CNN**
  - **CNN-LSTM**
  - **CNN-Attention-LSTM**
  - **CNN-Transformer**

## Fine-Tuning Phase

- Add a classifier on top of the encoder.
- Fine-tune the entire network using a limited labeled dataset.

## Experimentation Strategy 

Experiments (`train.py`) follow a repeated random split strategy (K repetitions of train/validation/test splitting):

1. Randomly split the full dataset into train, validation, and test sets.
2. Train the model on the training set.
3. Use the validation set to select the best-performing model.
4. Evaluate the selected model on the held-out test set.
5. Repeat the process K times with different random splits.
6. Report the average and confidence interval of test performance across K runs.
