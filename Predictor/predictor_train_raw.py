# predictor_train.py

import glob
import os
import time
from torch.nn.functional import pad
from torch.utils.data import TensorDataset, DataLoader
import re
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from Predictor.pytorchtools import EarlyStopping
from Predictor.predictor_models import *


torch.nn.utils.clip_grad_norm_


class PREDICT():
    def __init__(self,run_name='predictor_',dataset='SC_short', model_name = 'LSTMModel',
                 train_data_path = '/Data/SC/pred_train_exp_short.csv',
                 test_data_path = '/Data/SC/pred_test_exp_short.csv'):
        self.model_name = model_name
        self.patience = 20
        self.val_acc_list = []
        self.save_path = '/Predictor/results/model_SC_short/'
        self.dataset = dataset
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.seq1, self.exp = self.data_load(self.train_data_path)
        self.seq = self.seq_onehot(self.seq1)
        input_size = self.seq.shape[-1]
        self.input_size = input_size
        self.batch_size = 256
        # self.batch_size = 16
        self.hidden_size = 256
        self.conv_hidden = 128
        self.seq_len = 80
        # self.seq_len = 165
        self.output_size = 1
        self.lambda_l2 = 0.001
        self.dropout_rate = 0.2
        self.r = 161273
        # self.r = 12575
        torch.cuda.set_device(1)
        self.use_gpu = True if torch.cuda.is_available() else False
        self.build_model()
        self.checkpoint_dir = './checkpoint/' + run_name + '/'
        if not os.path.exists(self.checkpoint_dir): os.makedirs(self.checkpoint_dir)


    def data_load(self, data_path):
        data = open(data_path, 'r')
        next(data)
        seq = []
        exp = []
        for item in data:
            item = item.split(",")
            seq.append(item[0])
            exp.append(item[1])
        data.close()

        # Convert the expression data into array format
        expression = np.zeros((len(exp), 1))
        for i in range(len(exp)):
            expression[i] = float(exp[i])

        return seq, expression

    def data_load1(self, data_path):
        data = open(data_path, 'r')
        next(data)
        seq = []
        exp = []
        for item in data:
            item = item.split(",")
            seq.append(item[0])
            exp.append(item[1])
        data.close()

        expression = np.zeros((len(exp), 1))
        for i in range(len(exp)):
            expression[i] = float(exp[i])

        return seq, expression

    def string_to_array(self, my_string):
        my_string = my_string.lower()
        my_string = re.sub('[^acgt]', 'z', my_string)
        my_array = np.array(list(my_string))
        return my_array

    def one_hot_encode(self, my_array):
        label_encoder = LabelEncoder()
        label_encoder.fit(np.array(['a', 'c', 'g', 't', 'z']))
        integer_encoded = label_encoder.transform(my_array)
        onehot_encoder = OneHotEncoder(sparse=False, dtype=int)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        onehot_encoded = np.delete(onehot_encoded, -1, 1)
        return onehot_encoded

    def seq_onehot(self, seq):
        onehot_seq = [torch.tensor(self.one_hot_encode(self.string_to_array(s)), dtype=torch.float32) for s in seq]

        # Determine the maximum length
        max_length = max(matrix.shape[0] for matrix in onehot_seq)

        padded_tensor_list = []
        for matrix in onehot_seq:
            padding_length = max_length - matrix.shape[0]
            padded_tensor = pad(matrix, (0, 0, 0, padding_length), value=0)
            padded_tensor_list.append(padded_tensor)

        onehot_seq = torch.stack(padded_tensor_list, dim=0)

        return onehot_seq

    def build_model(self):
        if self.model_name == 'OnlyCNNModel':
            self.model = OnlyCNNModel(self.input_size, self.hidden_size, self.output_size, self.dropout_rate, self.lambda_l2)
        elif self.model_name == 'LSTMModel':
            self.model = LSTMModel(self.input_size, self.hidden_size, self.output_size, self.dropout_rate,
                                      self.lambda_l2)

        if self.use_gpu:
            self.model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=1e-3,weight_decay=self.lambda_l2)
        self.criterion = nn.MSELoss()

    def save_model(self, epoch):
        torch.save(self.model.state_dict(), self.checkpoint_dir + "model_weights_{}.pth".format(epoch))

    def load_model(self):
        '''
            Load model parameters from most recent epoch
        '''
        list_model = glob.glob(self.checkpoint_dir + "model*.pth")
        if len(list_model) == 0:
            print("[*] Checkpoint not found! Starting from scratch.")
            return 1 #file is not there
        chk_file = max(list_model, key=os.path.getctime)
        epoch_found = int( (chk_file.split('_')[-1]).split('.')[0])
        print("[*] Checkpoint {} found!".format(epoch_found))
        self.model.load_state_dict(torch.load(chk_file))
        return epoch_found


    def train(self):

        # Split training/validation and testing set
        expression = self.exp
        onehot_seq = self.seq

        seq = onehot_seq
        r = self.r
        train_feature = seq[0:r]
        train_label = expression[0:r]

        train_feature = torch.Tensor(train_feature)
        train_label = torch.Tensor(train_label)


        train_data = TensorDataset(train_feature, train_label)
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)


        # train model
        num_epochs = 100
        loss_total = 0.0
        best_loss = float('inf')

        clip_value = 1.0
        early_stopping = EarlyStopping(patience=self.patience, verbose=True,
                                       path=self.save_path + self.model_name + '_' + self.dataset + '.pth', stop_order='max')

        weighted_loss_function = WeightedMSELoss(weight=torch.Tensor([10.0]).cuda())

        with open('/Predictor/results/model_SC_short/onlyGRUaccuracy.txt', 'w') as f:
            for epoch in range(num_epochs):
                # self.model.train()
                epoch_loss = 0.0
                for batch_idx, (batch_feature, batch_label) in enumerate(train_loader):
                    batch_feature = batch_feature.cuda()
                    batch_label = batch_label.cuda()

                    self.optimizer.zero_grad()

                    with torch.cuda.amp.autocast():
                        output = self.model(batch_feature)
                        # loss = self.criterion(output, batch_label)
                        loss = weighted_loss_function(output, batch_label) # Use a weighted loss function

                    loss.backward()

                    nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)

                    self.optimizer.step()

                    epoch_loss += loss.item()

                avg_loss = epoch_loss / len(train_loader)
                print("Epoch:", epoch, "Loss:", avg_loss)
                if(avg_loss < best_loss):
                    best_loss = avg_loss
                    self.save_model(epoch)
                loss_total += avg_loss


                rho, cor, mse = self.evaluate()
                # f.write(f"Epoch {epoch}: Accuracy {rho}\n")
                f.write(f"rho:{rho},cor:{cor},mse:{mse}\n")
                early_stopping(val_loss=rho, model=self.model)
                if early_stopping.early_stop:
                    print('Early Stopping......')
                    break

    # Predict the entire dataset file
    def valdata1(self):
        self.model.load_state_dict(torch.load("./Predictor/results/model/LSTMModel_SC_256.pth"))

        with open("./Data/data_SC/Random_SC.txt", 'r') as f:
            next(f)
            inputseq = [line.strip() for line in f]


        valseq_onehot = self.seq_onehot(inputseq)
        valseq = torch.Tensor(valseq_onehot).cuda()

        with torch.no_grad():
            val_output = self.model(valseq)

        val_output = val_output.cpu()
        val_pred = val_output.numpy()

        # Write the results to a file
        with open("/Predictor/results/Random_SC.csv", "w") as f:
            for i, pred in enumerate(val_pred):
                f.write(inputseq[i] + "," + str(pred[0]) + "\n")

        return val_pred

if __name__ == '__main__':
    time_start = time.time()  # time record

    predict = PREDICT()
    predict.train()
    # predict.valdata1()

    time_end = time.time()
    time_sum = time_end - time_start
    print(time_sum)