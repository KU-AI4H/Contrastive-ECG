#train

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import pandas as pd
import numpy as np
import mlflow
from models import ECG_CNN_Encoder, ECG_Classifier



def main_train_test(pretrain_signals, pre_train_labels, data_df, embedded_size=64, kernel_size=15, dropout=0.1, CL_embedded_size=64, pretrain_num_epochs=50, num_epochs=70, print_results = False, save_results = False, plot_results = False, seed=42, cl_temp=0.07, augmentation1='dropout',  aug1_ratio=0.20, augmentation2='guassian_noise', aug2_ratio=0.15, domain ='time' , exp_name=None, encoder_name='CL_encoder', pre_batch_size=128, batch_size=12, pretrain_encoder_path=None, pretrain=True):

    set_seed(seed=seed)
    print(f'-------------------------domain = {domain}')

    # Step 1: Compute majority label per patient
    patient_labels = data_df.groupby('mrn')['Responder'].mean().round().astype(int)

    # Step 2: First split: Train vs (Validation + Test)
    train_patients, temp_patients = train_test_split(
        patient_labels.index,
        test_size=0.3,  # 20% goes to (val + test)
        stratify=patient_labels,
        random_state=seed
    )

    # Step 3: Second split: Validation vs Test
    temp_labels = patient_labels.loc[temp_patients]
    val_patients, test_patients = train_test_split(
        temp_labels.index,
        test_size=0.5,  # Split remaining 30% into 15% val and 15% test
        stratify=temp_labels,
        random_state=seed
    )

    # Step 4: Assign ECGs based on patient split
    train_df = data_df[data_df['mrn'].isin(train_patients)]
    val_df = data_df[data_df['mrn'].isin(val_patients)]
    test_df = data_df[data_df['mrn'].isin(test_patients)]

    # (Optional) If you want to drop duplicate ECGs for each patient (keep first)
    val_df = val_df.drop_duplicates(subset='mrn', keep='first')
    test_df = test_df.drop_duplicates(subset='mrn', keep='first')


    #############################
    # # Splitting index while preserving label distribution
    # train_idx, test_idx = train_test_split(
    #     data_df.index, test_size=0.15, random_state=seed, stratify=data_df['Responder']
    # )

    # # Creating train and test DataFrames
    # train_df = data_df.loc[train_idx].copy()
    # test_df = data_df.loc[test_idx].copy()
    #############################



    max_amp = 0
    for i in range(train_df.shape[0]):
        for j in range(12):
            max_i = train_df.iloc[i, j].max()
            if max_i > max_amp:
                max_amp = max_i

    print('max_amp:', max_amp)
    # Preprocess train, val, and test
    X_train, y_train = zip(*train_df.apply(preprocessing, axis=1, args=(max_amp,)))
    X_val, y_val = zip(*val_df.apply(preprocessing, axis=1, args=(max_amp,)))
    X_test, y_test = zip(*test_df.apply(preprocessing, axis=1, args=(max_amp,)))

    # Create Datasets
    train_dataset = ECGDataset(X_train, y_train)
    val_dataset = ECGDataset(X_val, y_val)
    test_dataset = ECGDataset(X_test, y_test)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


    # Initialize device, model, criterion, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device: {device}')
    if domain == 'time':
        signal_length=5000
    else:
        signal_length=2500

    # Initialize dictionaries to store metrics
    metrics = {
        'epoch': [],'epoch_cl': [],
        'train_loss': [], 'cl_loss':[], 'train_accuracy': [], 'train_precision': [],
        'train_recall': [], 'train_f1': [], 'train_auc': [],'train_pr_auc': [],
        'val_loss': [], 'val_accuracy': [], 'val_precision': [],
        'val_recall': [], 'val_f1': [], 'val_auc': [], 'val_pr_auc': [],
        'test_loss': [], 'test_accuracy': [], 'test_precision': [],
        'test_recall': [], 'test_f1': [], 'test_auc': [], 'test_pr_auc': []
    }
    model_name = 'ECG_CNN_1D' 
    save_folder = os.path.join(os.getcwd(), model_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    best_val_auc = -1.0
    best_metrics = {}
        
    encoder = ECG_Encoder(signal_length=signal_length, embedded_size=embedded_size, CL_embedded_size=CL_embedded_size, kernel_size=kernel_size, dropout=dropout, seed=seed).to(device)

    if pretrain: 
        print('pretraining started ...')

        if pretrain_encoder_path==None:
            pretrain_dataset = ECGDataset(pretrain_signals, pre_train_labels)
            pretrain_loader = DataLoader(pretrain_dataset, batch_size=pre_batch_size, shuffle=True, num_workers=4)
            encoder_path = f"{encoder_name}_aug1_{augmentation1}_ratio_{aug1_ratio}_agu2_{augmentation2}_ratio_{aug2_ratio}_domain_{domain}_epoch_{pretrain_num_epochs}_num_{len(pretrain_signals)}_batchsize_{pre_batch_size}_clemb_{CL_embedded_size}_cltemp_{cl_temp}.pth"
            print(f'encoder path: {encoder_path}')
            print()
        else:
            encoder_path = pretrain_encoder_path

        if os.path.exists(encoder_path):
            print("Loading pre-trained encoder...")
            encoder.load_state_dict(torch.load(encoder_path, map_location=device))
            encoder.train() 
            metrics[f'cl_loss'] = [0] * pretrain_num_epochs

        else:
            # Define learning rates
            pretrain_lr = 0.00001 # Learning rate for pretraining
            # Initialize optimizers
            pretrain_optimizer = optim.Adam(encoder.parameters(), lr=pretrain_lr, weight_decay=1e-5)
            cl_loss_fn = NTXentLoss(temperature=cl_temp)
            
            print()
            print('Unsupervised Pre-Training started...')
            for epoch in tqdm(range(pretrain_num_epochs)):
                print()
                encoder.train() 
                cl_loss = 0.0
                for signals, _ in pretrain_loader:
                    signals = signals.to(device)
                    if domain != 'time':
                        signal_length = signals.shape[-1]
                        _, signals = ecg_to_frequency_domain(signals)
                        signals = signals[:, :, :signal_length // 2]
                        #signals = preprocess_magnitude_batch(signals, method='minmax')

                    pretrain_optimizer.zero_grad()

                    if augmentation1 == 'dropout':
                        aug_signal1 = dropout_augmentation(signals, zero_percent=aug1_ratio, seed=seed)
                    elif augmentation1 == 'guassian_noise':
                        aug_signal1 =gaussian_noise_augmentation(signals, noise_factor=aug1_ratio, seed=seed)
                    elif augmentation1 == 'zero-masking':
                        aug_signal1 =zero_masking_augmentation(signals, ratio=aug1_ratio, seed=seed)
                    elif augmentation1 == 'permutation':
                        aug_signal1 =permutation_augmentation(signals, m=aug1_ratio, seed=seed)
                    elif augmentation1 == "time-wraping":
                        aug_signal1 = time_wrap_segments_augmentation(signals, m=aug1_ratio[0], warp_percent=aug1_ratio[1], seed=None)

                    if augmentation2 == 'dropout':
                        aug_signal2 = dropout_augmentation(signals, zero_percent=aug2_ratio, seed=seed)
                    elif augmentation2 == 'guassian_noise':
                        aug_signal2 =gaussian_noise_augmentation(signals, noise_factor=aug2_ratio, seed=seed)
                    elif augmentation2 == 'zero-masking':
                        aug_signal2 =zero_masking_augmentation(signals, ratio=aug2_ratio, seed=seed)
                    elif augmentation2 == 'permutation':
                        aug_signal2 =permutation_augmentation(signals, m=aug2_ratio, seed=seed)
                    elif augmentation2 == "time-wraping":
                        aug_signal2 = time_wrap_segments_augmentation(signals, m=aug2_ratio[0], warp_percent=aug2_ratio[1], seed=None)

                    _, aug_repr1 = encoder(aug_signal1)
                    _, aug_repr2 = encoder(aug_signal2)
                    contrastive_loss = cl_loss_fn(aug_repr1, aug_repr2)
                    contrastive_loss.backward()
                    pretrain_optimizer.step()
                    cl_loss += contrastive_loss.item()

                metrics[f'cl_loss'].append(cl_loss / len(pretrain_loader))

                if print_results:
                    print(f"Epoch [{epoch + 1}/{pretrain_num_epochs}], Train Loss: {cl_loss / len(pretrain_loader):.4f}")

            torch.save(encoder.state_dict(), encoder_path)
    else:
        print('No pretraining ...')
        metrics[f'cl_loss'] = [0] * pretrain_num_epochs


    classifier = ECG_Classifier(encoder, embedded_size=embedded_size, dropout=dropout).to(device)
    train_lr = 0.0001      # Learning rate for supervised training
    optimizer = optim.Adam(classifier.parameters(), lr=train_lr, weight_decay=1e-4)
    criterion = nn.BCELoss()

    best_classifier_state = None

    print()
    print('Supervised Training started...')
    #model.freeze_encoder(freeze=True)
    ##encoder.eval()
    for epoch in tqdm(range(num_epochs)):
        print()
        classifier.train() 
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        for signals, labels in train_loader:
            signals, labels = signals.to(device), labels.to(device).unsqueeze(1)
            if domain != 'time':
                signal_length = signals.shape[-1]
                _, signals = ecg_to_frequency_domain(signals)
                signals = signals[:, :, :signal_length // 2]
                #signals = preprocess_magnitude_batch(signals, method='minmax')

            optimizer.zero_grad()
            outputs = classifier(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(outputs.detach().cpu().numpy())
            train_labels.extend(labels.detach().cpu().numpy())

        record_metrics('train', train_loss, train_preds, train_labels, metrics, epoch, len(train_loader))

        classifier.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            #for signals, labels in tqdm(test_loader, desc=f"Validating Epoch {epoch + 1}/{num_epochs}"):
            for signals, labels in val_loader:
                signals, labels = signals.to(device), labels.to(device).unsqueeze(1)
                if domain != 'time':
                    _, signals = ecg_to_frequency_domain(signals)
                    signals = signals[:, :, :signal_length // 2]
                    #signals = preprocess_magnitude_batch(signals, method='zscore')
                outputs = classifier(signals)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_preds.extend(outputs.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        record_metrics('val', val_loss, val_preds, val_labels, metrics, epoch, len(test_loader))

        if print_results:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {metrics['train_loss'][-1]:.4f},  CL Loss: {metrics['cl_loss'][-1]:.4f}, "
                f"Train Accuracy: {metrics['train_accuracy'][-1]:.4f}, Train Precision: {metrics['train_precision'][-1]:.4f}, "
                f"Train Recall: {metrics['train_recall'][-1]:.4f}, Train F1: {metrics['train_f1'][-1]:.4f}, Train AUC: {metrics['train_auc'][-1]:.4f}, Train PRAUC: {metrics['train_pr_auc'][-1]:.4f}, "
                f"Val Loss: {metrics['val_loss'][-1]:.4f}, Val Accuracy: {metrics['val_accuracy'][-1]:.4f}, "
                f"Val Precision: {metrics['val_precision'][-1]:.4f}, Val Recall: {metrics['val_recall'][-1]:.4f}, "
                f"Val F1: {metrics['val_f1'][-1]:.4f}, Val AUC: {metrics['val_auc'][-1]:.4f}, Val PRAUC: {metrics['val_pr_auc'][-1]:.4f}")


        # Check if this model has the best validation AUC
        if metrics['val_auc'][-1] > best_val_auc:
        # if metrics['val_pr_auc'][-1] > best_val_auc:
            best_val_auc = metrics['val_auc'][-1]
            #best_val_auc = metrics['val_pr_auc'][-1]
            best_metrics = {
                'epoch': epoch + 1,
                'train_loss': metrics['train_loss'][-1],
                'train_accuracy': metrics['train_accuracy'][-1],
                'train_precision': metrics['train_precision'][-1],
                'train_recall': metrics['train_recall'][-1],
                'train_f1': metrics['train_f1'][-1],
                'train_auc': metrics['train_auc'][-1],
                'train_pr_auc': metrics['train_pr_auc'][-1],
                'val_loss': metrics['val_loss'][-1],
                'val_accuracy': metrics['val_accuracy'][-1],
                'val_precision': metrics['val_precision'][-1],
                'val_recall': metrics['val_recall'][-1],
                'val_f1': metrics['val_f1'][-1],
                'val_auc': metrics['val_auc'][-1],
                'val_pr_auc': metrics['val_pr_auc'][-1]
            }
            best_classifier_state = classifier.state_dict()
            #torch.save(model.state_dict(), os.path.join(save_folder, 'best_model.pth'))

    # End of Training

    # Load the best model weights for final testing
    classifier.load_state_dict(best_classifier_state)

    # Evaluate on test set
    classifier.eval()
    test_loss = 0.0
    test_preds = []
    test_labels = []

    with torch.no_grad():
        for signals, labels in test_loader:
            signals, labels = signals.to(device), labels.to(device).unsqueeze(1)
            if domain != 'time':
                signal_length = signals.shape[-1]
                _, signals = ecg_to_frequency_domain(signals)
                signals = signals[:, :, :signal_length // 2]
            outputs = classifier(signals)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            test_preds.extend(outputs.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    # Optionally, record final test metrics
    record_metrics('test', test_loss, test_preds, test_labels, metrics, epoch, len(test_loader))
    best_test_metrics = {
                'test_loss': metrics['test_loss'][-1],
                'test_accuracy': metrics['test_accuracy'][-1],
                'test_precision': metrics['test_precision'][-1],
                'test_recall': metrics['test_recall'][-1],
                'test_f1': metrics['test_f1'][-1],
                'test_auc': metrics['test_auc'][-1],
                'test_pr_auc': metrics['test_pr_auc'][-1]
            }



    # Convert metrics to DataFrame
    metrics['epoch'] = list(range(1, num_epochs + 1))
    metrics['epoch_cl'] = list(range(1, pretrain_num_epochs + 1))
    #metrics_df = pd.DataFrame(metrics)
    metrics_df = metrics



    # Save metrics and plots
    if save_results:
        save_metrics(metrics_df, f'{exp_name}_seed={seed}', save_folder)



    # Print the best model's performance metrics
    if print_results:
        print("\nBest Model Performance Metrics:")
        print(f"Epoch: {best_metrics['epoch']}")
        print(f"Train Loss: {best_metrics['train_loss']:.4f}, Train Accuracy: {best_metrics['train_accuracy']:.4f}, "
            f"Train Precision: {best_metrics['train_precision']:.4f}, Train Recall: {best_metrics['train_recall']:.4f}, "
            f"Train F1: {best_metrics['train_f1']:.4f}, Train AUC: {best_metrics['train_auc']:.4f}")
        print(f"Val Loss: {best_metrics['val_loss']:.4f}, Val Accuracy: {best_metrics['val_accuracy']:.4f}, "
            f"Val Precision: {best_metrics['val_precision']:.4f}, Val Recall: {best_metrics['val_recall']:.4f}, "
            f"Val F1: {best_metrics['val_f1']:.4f}, Val AUC: {best_metrics['val_auc']:.4f}, Val PRAUC: {best_metrics['val_pr_auc']:.4f}")
        print(f"test Loss: {best_test_metrics['test_loss']:.4f}, test Accuracy: {best_test_metrics['test_accuracy']:.4f}, "
            f"test Precision: {best_test_metrics['test_precision']:.4f}, test Recall: {best_test_metrics['test_recall']:.4f}, "
            f"test F1: {best_test_metrics['test_f1']:.4f}, test AUC: {best_test_metrics['test_auc']:.4f}, test PRAUC: {best_test_metrics['test_pr_auc']:.4f}")
    

    # Plotting example
    if plot_results:
        plot_metrics(metrics_df, 'loss', model_name, save_folder)
        plot_metrics(metrics_df, 'cl_loss', model_name, save_folder)
        plot_metrics(metrics_df, 'accuracy', model_name, save_folder)
        plot_metrics(metrics_df, 'precision', model_name, save_folder)
        plot_metrics(metrics_df, 'recall', model_name, save_folder)
        plot_metrics(metrics_df, 'f1', model_name, save_folder)
        plot_metrics(metrics_df, 'auc', model_name, save_folder)
        plot_metrics(metrics_df, 'pr_auc', model_name, save_folder)

    return best_metrics, best_test_metrics


def cross_val_main(pretrain_signals, pre_train_labels, data_df, print_seed_results=True, record_seed_results=True, pretrain_num_epochs=50, num_epochs=60, embedded_size=64, kernel_size=15, dropout = 0.1, CL_embedded_size=64, domain='time', cl_temp=None,  augmentation1='dropout',  aug1_ratio=0.20, augmentation2='guassian_noise', aug2_ratio=0.15,  encoder_name='CL_encoder', pre_batch_size=128, batch_size=128, pretrain_encoder_path=None, pretrain=True):

    print(f'domain is {domain}')
    best_metrics_dict = {'epoch':[], 'test_accuracy': [], 'test_precision': [], 'test_recall': [], 'test_precision':[], 'test_recall':[], 'test_f1': [], 'test_auc': [], 'test_pr_auc':[]}
    #seeds= [42, 123, 51, 10, 12]
    #seeds= [42, 123, 51, 15, 12]
    seeds= [321, 123, 51, 15, 12, 1234, 999]
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)


        best_metrics, best_test_metrics = main_train_test(pretrain_signals, pre_train_labels, data_df, embedded_size=embedded_size, kernel_size=kernel_size, dropout=dropout, CL_embedded_size=CL_embedded_size, pretrain_num_epochs=pretrain_num_epochs, num_epochs=num_epochs, print_results = False, save_results = False, plot_results = False, seed=seed, cl_temp=cl_temp,  augmentation1=augmentation1,  aug1_ratio=aug1_ratio, augmentation2=augmentation2, aug2_ratio=aug2_ratio, domain =domain , exp_name=None, encoder_name=encoder_name, pre_batch_size=pre_batch_size, batch_size=batch_size, pretrain_encoder_path=pretrain_encoder_path, pretrain=pretrain)

        print(f'----seed={seed} is done')
        best_metrics_dict['epoch'].append(best_metrics['epoch'])
        best_metrics_dict['test_accuracy'].append(best_test_metrics['test_accuracy'])
        best_metrics_dict['test_precision'].append(best_test_metrics['test_precision'])
        best_metrics_dict['test_recall'].append(best_test_metrics['test_recall'])
        best_metrics_dict['test_f1'].append(best_test_metrics['test_f1'])
        best_metrics_dict['test_auc'].append(best_test_metrics['test_auc'])
        best_metrics_dict['test_pr_auc'].append(best_test_metrics['test_pr_auc'])
        print(best_metrics)
        print(best_test_metrics)
        print('---------------------------------------------------------')


    if print_seed_results:
        for metric in best_metrics_dict.keys():
            print()
            print(f'mean {metric}:', np.mean(best_metrics_dict[metric]))
            print(f'max {metric}:', np.max(best_metrics_dict[metric]))
            print(f'min {metric}:', np.min(best_metrics_dict[metric]))
            print(f'CI {metric}:', calculate_confidence_interval_error(best_metrics_dict[metric]))
            print()

    if record_seed_results:
        model_name = 'ECG_CNN_1D'  # Change this to your model name
        file_name = f'{encoder_name}_aug1_{augmentation1}_ratio_{aug1_ratio}_agu2_{augmentation2}_ratio_{aug2_ratio}_domain_{domain}_epoch_{pretrain_num_epochs}_num_pre_{len(pretrain_signals)}_prebatchsize_{pre_batch_size}_clemb_{CL_embedded_size}_cltemp_{cl_temp}_emb_{embedded_size}_num_finetune_data_{len(data_df)}.txt'
        save_folder = os.path.join(os.getcwd(), model_name, 'results')
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        with open(os.path.join(save_folder, file_name), 'w') as file:
            for metric in best_metrics_dict.keys():
                file.write('\n')
                file.write(f'mean {metric}: {np.mean(best_metrics_dict[metric])}\n')
                file.write(f'max {metric}: {np.max(best_metrics_dict[metric])}\n')
                file.write(f'min {metric}: {np.min(best_metrics_dict[metric])}\n')
                file.write(f'CI {metric}: {calculate_confidence_interval_error(best_metrics_dict[metric])}\n')
                file.write('\n')

    final_metric_dict = {'test_accuracy':np.mean(best_metrics_dict['test_accuracy']), 
                         'test_precision': np.mean(best_metrics_dict['test_precision']), 
                         'test_recall': np.mean(best_metrics_dict['test_recall']), 
                         'test_f1': np.mean(best_metrics_dict['test_f1']), 
                         'test_auc': np.mean(best_metrics_dict['test_auc']),
                         'test_pr_auc': np.mean(best_metrics_dict['test_pr_auc']),
                         'epoch':np.mean(best_metrics_dict['epoch'])}

    return final_metric_dict





# load pretraining unlabled data
train_files = glob.glob('/Volumes/research/ecg_echo/echo_data/train_abnormal_ecgs/*')#[1:25]
test_files = glob.glob('/Volumes/research/ecg_echo/echo_data/test_abnormal_ecgs/*')#[1:25]

tfrecord_files = train_files
pretrain_signals = []
pre_train_labels = []
raw_dataset = tf.data.TFRecordDataset(tfrecord_files)
for raw_record in raw_dataset:
    signal, label = parse_tfr_element(raw_record)
    pretrain_signals.append(signal)
    pre_train_labels.append(label)

tfrecord_files = test_files
raw_dataset = tf.data.TFRecordDataset(tfrecord_files)
for raw_record in raw_dataset:
    signal, label = parse_tfr_element(raw_record)
    pretrain_signals.append(signal)
    pre_train_labels.append(label)



mlflow.sklearn.autolog(disable=True)

file_path = '/Volumes/research/ecg_echo/echo_data/PRE_CRT_ecgs_processed.pkl'
#PRE_CRT_ecgs_processed
#CRT_data_processed
#CRT_ecgs_processed_4
with open(file_path, 'rb') as f:
    data_df = pickle.load(f)

#ata_df = pd.read_csv('/Volumes/research/ecg_echo/echo_data/CRT_data_processed_fixed.csv', index_col=0)
print(data_df.shape)
#data_df = data_df[data_df['full_error_processing']==False]
print(data_df.shape)

data_df['DOS'] = pd.to_datetime(data_df['DOS'])
# Check the type of data
# data_df.sort_values(by=['mrn', 'DOS'], inplace=True)
data_df.sort_values(by=['mrn', 'DOS'], ascending=False, inplace=True)
# data_df.drop_duplicates(subset=['PatientNum'], keep='first', inplace=True)
print(data_df.shape)
print(f'num unique patients {data_df.mrn.nunique()}')
data_df = data_df.dropna(subset=['Responder'])

print(data_df.shape)
print(f'num unique patients {data_df.mrn.nunique()}')

print(f'lable ratio {data_df[data_df["Responder"] == 1].shape[0] / data_df.shape[0]}')


# Check if any patient has multiple labels
mixed_label_patients = data_df.groupby('mrn')['Responder'].nunique()
mixed_label_patients = mixed_label_patients[mixed_label_patients > 1]
print()
# Print problematic patients
print("Patients with mixed labels:\n", mixed_label_patients)
print()

data_df = data_df[['full_lead_I', 'full_lead_II', 'full_lead_III', 'full_lead_AVR', 'full_lead_AVL', 'full_lead_AVF', 'full_lead_V1', 'full_lead_V2', 'full_lead_V3', 'full_lead_V4', 'full_lead_V5', 'full_lead_V6', 'Responder', 'mrn']]

final_metric_dict = cross_val_main(pretrain_signals, pre_train_labels, data_df, print_seed_results=True, record_seed_results=False,  pretrain_num_epochs=50, num_epochs=100, embedded_size=256, kernel_size=15, dropout = 0.3, CL_embedded_size=128, domain='time', cl_temp=0.07,  augmentation1='guassian_noise',  aug1_ratio=0.05, augmentation2='guassian_noise', aug2_ratio=0.25,  encoder_name='CL_CNN_encoder', pre_batch_size=150, batch_size=128, pretrain_encoder_path=None, pretrain=True)
