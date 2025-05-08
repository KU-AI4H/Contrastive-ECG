# Self-Supervised Pretraining of ECG Signals Using Contrastive Learning

This repository implements **self-supervised contrastive learning** for 12-lead ECG signals using various augmentation strategies and encoder architectures. After pretraining on massive unlabeled ECG data, the learned encoder is fine-tuned for a downstream **binary classification** task.

---

## 🧪 Pretraining Phase

- Self-supervised contrastive learning is performed using positive pairs from augmented views of the same ECG signal.
- Training is done on a large **unlabeled ECG dataset** using NT-Xent (or similar) loss to bring representations of similar signals closer and dissimilar ones apart.

---

## 🔁 Contrastive Augmentation Strategies

Defined in `CL_augmentations.py`, these augmentations provide diverse views of the same ECG signal:

- **Time Wrapping**: Alternating segments of the ECG are stretched or compressed to simulate temporal warping.
- **Permutation**: ECG signals are split into `m` segments and randomly shuffled.
- **Zero Masking**: Consecutive portions of the ECG are set to zero.
- **Dropout Masking**: Randomly zeros out 10% of signal values per lead in each batch.
- **Gaussian Noise**: Adds noise scaled to signal magnitude for robustness.
- **CLOCKS Augmentation**: Implements spatial, temporal, and patient-level contrast based on [CLOCS (Kiyasseh et al., ICML 2021)](https://proceedings.mlr.press/v139/kiyasseh21a.html).

---

## 🧠 Encoder Architectures

Implemented in `models.py`, multiple encoders are supported to extract meaningful ECG representations:

-  **CNN** – Temporal filters for local pattern learning.
-  **CNN-LSTM** – Combines convolution with temporal memory.
-  **CNN-Attention-LSTM** – Adds attention over LSTM outputs.
-  **CNN-Transformer** – Combines convolutional front-end with self-attention layers.

---

## 🔄 Fine-Tuning Phase

After pretraining:
- A **classifier** is added on top of the pretrained encoder.
- The full model is **fine-tuned end-to-end** using a limited labeled ECG dataset.

---

## 📊 Experimentation Strategy

Implemented in `train.py`, all experiments follow a **repeated random sub-sampling protocol**:

1. Randomly split patients into train, validation, and test sets.
2. Train the model on the training set.
3. Use validation performance to select the best checkpoint.
4. Evaluate on the held-out test set.
5. Repeat the full process **K times** with different seeds.
6. Report **mean ± confidence interval** for test performance.

---


