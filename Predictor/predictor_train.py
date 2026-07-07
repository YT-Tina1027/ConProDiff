import sys
import os
import pickle
import time
import re
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import pad
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

plt.switch_backend('Agg')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Predictor.pytorchtools import EarlyStopping
from Predictor.predictor_models import LSTMModel, WeightedMSELoss


class PREDICT():
    def __init__(self, target_dataset='EC'):

        # --- 1. 基础配置 ---
        self.path_config = {
            # ✅ 统一用 ecoli_exp_all.csv，与 diffusion 训练数据一致
            'EC': '/home/yt/Code/DNA-Diffusion/Data/ecoli_exp_all.csv',
            'SC': '/home/yt/Code/DNA-Diffusion/Data/SC_exp_short.csv',
        }

        if target_dataset not in self.path_config:
            raise ValueError(f"Dataset {target_dataset} not found.")

        self.dataset_name = target_dataset
        self.data_path    = self.path_config[target_dataset]
        self.save_path    = f'/home/yt/Code/DNA-Diffusion/Predictor/results/model_{target_dataset}/'

        os.makedirs(self.save_path, exist_ok=True)

        print(f"[*] Loading Data for : {self.dataset_name}")
        print(f"[*] Data path        : {self.data_path}")

        # --- 2. 数据处理 ---
        raw_seqs, raw_exps = self.data_load(self.data_path)

        print(f"[*] Total samples loaded: {len(raw_seqs)}")

        # ✅ fit scaler 并保留引用，训练结束后保存到磁盘
        self.scaler = StandardScaler()
        self.scaler.fit(raw_exps)
        normalized_exps = self.scaler.transform(raw_exps)

        print(f"[*] Scaler: mean={self.scaler.mean_[0]:.4f}, std={self.scaler.scale_[0]:.4f}")

        all_seq_tensor = self.seq_onehot(raw_seqs)
        all_exp_tensor = torch.Tensor(normalized_exps)

        self.seq_len       = all_seq_tensor.shape[1]
        self.input_size    = all_seq_tensor.shape[-1]
        self.total_samples = len(all_seq_tensor)

        print(f"[*] Samples: {self.total_samples}, Seq len: {self.seq_len}, Input size: {self.input_size}")

        # --- 3. 划分数据 9:1，固定随机种子 ---
        X_train, X_test, y_train, y_test = train_test_split(
            all_seq_tensor,
            all_exp_tensor,
            test_size=0.1,      # ✅ 改为 9:1
            random_state=42,
        )

        print(f"[*] Train: {len(X_train)}, Test: {len(X_test)}")

        self.train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=256,
            shuffle=True,
        )
        self.test_loader = DataLoader(
            TensorDataset(X_test, y_test),
            batch_size=256,
            shuffle=False,
        )

        # --- 4. 模型配置 ---
        self.model_name   = 'LSTMModel'
        self.hidden_size  = 256
        self.output_size  = 1
        self.dropout_rate = 0.2
        self.lambda_l2    = 0.001
        self.patience     = 50

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Device: {self.device}")

        self.build_model()

    # ------------------------------------------------------------------
    # 数据加载
    # ------------------------------------------------------------------

    def data_load(self, data_path):
        seq = []
        exp = []
        with open(data_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) >= 2:
                    try:
                        value = float(parts[1])
                        seq.append(parts[0])
                        exp.append(value)
                    except ValueError:
                        # 跳过表头行
                        continue
        return seq, np.array(exp).reshape(-1, 1)

    # ------------------------------------------------------------------
    # 序列编码
    # ------------------------------------------------------------------

    def string_to_array(self, my_string):
        my_string = my_string.lower()
        my_string = re.sub('[^acgt]', 'z', my_string)
        return np.array(list(my_string))

    def one_hot_encode(self, my_array):
        mapping = {'a': 0, 'c': 1, 'g': 2, 't': 3}
        onehot_encoded = np.zeros((len(my_array), 4), dtype=int)
        for i, char in enumerate(my_array):
            if char in mapping:
                onehot_encoded[i, mapping[char]] = 1
        return onehot_encoded

    def seq_onehot(self, seq_list):
        tensor_list = [
            torch.tensor(
                self.one_hot_encode(self.string_to_array(s)),
                dtype=torch.float32,
            )
            for s in seq_list
        ]
        max_length = max(t.shape[0] for t in tensor_list)
        padded_list = []
        for t in tensor_list:
            pad_len = max_length - t.shape[0]
            padded_t = pad(t, (0, 0, 0, pad_len), value=0)
            padded_list.append(padded_t)
        return torch.stack(padded_list, dim=0)

    # ------------------------------------------------------------------
    # 模型构建
    # ------------------------------------------------------------------

    def build_model(self):
        self.model = LSTMModel(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            dropout_rate=self.dropout_rate,
            lambda_l2=self.lambda_l2,
        )
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-3,
            weight_decay=self.lambda_l2,
        )

        try:
            self.criterion = WeightedMSELoss(
                weight=torch.Tensor([10.0]).to(self.device)
            )
        except Exception:
            self.criterion = nn.MSELoss()

    # ------------------------------------------------------------------
    # 评估
    # ------------------------------------------------------------------

    def evaluate(self, dataloader):
        self.model.eval()
        preds   = []
        targets = []
        total_loss = 0.0

        with torch.no_grad():
            for feature, label in dataloader:
                feature = feature.to(self.device)
                label   = label.to(self.device)
                output  = self.model(feature)

                loss = self.criterion(output, label)
                total_loss += loss.item()

                preds.extend(output.cpu().numpy().flatten())
                targets.extend(label.cpu().numpy().flatten())

        avg_loss = total_loss / len(dataloader)
        preds    = np.array(preds)
        targets  = np.array(targets)
        mse      = mean_squared_error(targets, preds)
        rho, _   = pearsonr(targets, preds)
        cor, _   = spearmanr(targets, preds)

        self.model.train()
        return avg_loss, rho, cor, mse

    # ------------------------------------------------------------------
    # 画图
    # ------------------------------------------------------------------

    def plot_training_curves(self, history):
        epochs = range(len(history['train_loss']))

        plt.close('all')
        plt.figure(figsize=(18, 5))

        plt.subplot(1, 3, 1)
        plt.plot(epochs, history['train_loss'],    'b-',  label='Train Loss')
        plt.plot(epochs, history['val_loss'],      'r--', label='Val Loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(epochs, history['train_pearson'], 'b-',  label='Train Pearson')
        plt.plot(epochs, history['val_pearson'],   'r--', label='Val Pearson')
        plt.title('Pearson Correlation')
        plt.xlabel('Epochs')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.plot(epochs, history['train_mse'],     'b-',  label='Train MSE')
        plt.plot(epochs, history['val_mse'],       'r--', label='Val MSE')
        plt.title('MSE')
        plt.xlabel('Epochs')
        plt.legend()
        plt.grid(True)

        save_file = self.save_path + f'{self.model_name}_{self.dataset_name}_training_curves.png'

        os.makedirs(os.path.dirname(save_file), exist_ok=True)  # ← 加这一行

        plt.tight_layout()
        plt.savefig(save_file)
        plt.close('all')

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------

    def train(self):
        print(f"[*] Start Training {self.model_name} on {self.dataset_name}...")

        best_model_path = self.save_path + f'{self.model_name}_{self.dataset_name}_best.pth'

        early_stopping = EarlyStopping(
            patience=self.patience,
            verbose=True,
            path=best_model_path,
            stop_order='max',
        )

        num_epochs = 200
        clip_value = 1.0

        history = {
            'train_loss':    [],
            'val_loss':      [],
            'train_pearson': [],
            'val_pearson':   [],
            'train_mse':     [],
            'val_mse':       [],
        }

        log_file_path = self.save_path + 'training_log.txt'

        with open(log_file_path, 'w') as f:
            for epoch in range(num_epochs):

                # --- 训练阶段 ---
                train_loss_accum = 0.0
                train_preds      = []
                train_targets    = []

                for batch_feature, batch_label in self.train_loader:
                    batch_feature = batch_feature.to(self.device)
                    batch_label   = batch_label.to(self.device)

                    self.optimizer.zero_grad()

                    with torch.amp.autocast(
                        device_type=self.device.type,
                        enabled=(self.device.type == "cuda"),
                    ):
                        output = self.model(batch_feature)
                        loss   = self.criterion(output, batch_label)

                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
                    self.optimizer.step()

                    train_loss_accum += loss.item()
                    train_preds.extend(output.detach().cpu().numpy().flatten())
                    train_targets.extend(batch_label.detach().cpu().numpy().flatten())

                # --- 训练集指标 ---
                train_avg_loss = train_loss_accum / len(self.train_loader)
                train_preds    = np.array(train_preds)
                train_targets  = np.array(train_targets)
                train_mse      = mean_squared_error(train_targets, train_preds)
                train_rho, _   = pearsonr(train_targets, train_preds)

                # --- 验证阶段 ---
                val_loss, val_rho, val_cor, val_mse = self.evaluate(self.test_loader)

                # --- 记录 ---
                history['train_loss'].append(train_avg_loss)
                history['val_loss'].append(val_loss)
                history['train_pearson'].append(train_rho)
                history['val_pearson'].append(val_rho)
                history['train_mse'].append(train_mse)
                history['val_mse'].append(val_mse)

                log_str = (
                    f"Epoch: {epoch} | "
                    f"Train Loss: {train_avg_loss:.4f} | Val Loss: {val_loss:.4f} | "
                    f"Train Pearson: {train_rho:.4f} | Val Pearson: {val_rho:.4f} | "
                    f"Train MSE: {train_mse:.4f} | Val MSE: {val_mse:.4f}"
                )
                print(log_str)
                f.write(log_str + "\n")

                self.plot_training_curves(history)

                early_stopping(val_loss=val_rho, model=self.model)
                if early_stopping.early_stop:
                    print('[!] Early Stopping triggered.')
                    break

        # --- 最终评估 ---
        print("\n" + "=" * 50)
        print("[*] Loading Best Model for Final Evaluation...")
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        val_loss, final_rho, final_cor, final_mse = self.evaluate(self.test_loader)

        print(f"FINAL REPORT ({self.dataset_name}):")
        print(f"   Pearson (r) : {final_rho:.4f}")
        print(f"   Spearman (ρ): {final_cor:.4f}")
        print(f"   MSE         : {final_mse:.4f}")
        print("=" * 50)

        # ✅ 保存 scaler，供 diffusion 评估时 inverse transform 使用
        scaler_path = self.save_path + f'scaler_{self.dataset_name}.pkl'
        with open(scaler_path, "wb") as sf:
            pickle.dump(self.scaler, sf)

        print(f"[*] Scaler saved → {scaler_path}")
        print(f"    mean={self.scaler.mean_[0]:.4f}, std={self.scaler.scale_[0]:.4f}")


if __name__ == '__main__':
    TARGET_DATASET = 'SC'
    start_time = time.time()
    predictor = PREDICT(target_dataset=TARGET_DATASET)
    predictor.train()
    print(f"Total Time: {time.time() - start_time:.2f} seconds")