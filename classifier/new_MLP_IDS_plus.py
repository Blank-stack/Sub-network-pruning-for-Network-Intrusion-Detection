import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from prune.GateLayer import GateMLP
from torch import optim
from my_tools import *
from tqdm import tqdm
from prune.Loss import DebiasedSupConLoss

warnings.filterwarnings("ignore")


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self, logits, targets, weights=None):
        if len(targets.shape) == 2: targets = torch.argmax(targets, dim=1)
        targets = targets.long().to(self.device)
        if weights is None:
            weights = np2ts(np.ones(len(targets)))
        # CE loss
        log_probs = F.log_softmax(logits, dim=-1)
        # Gather log probabilities for the target classes
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze()

        # Multiply by weights and compute the final loss
        weighted_loss = -weights * target_log_probs
        return weighted_loss.mean()


# from training import loss  # GeneralizedCELoss
class GeneralizedCELoss(nn.Module):
    def __init__(self, q=0.7):
        super(GeneralizedCELoss, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.q = q

    def forward(self, logits, targets, ws=None):
        if len(targets.shape) == 2: targets = torch.argmax(targets, dim=1)
        targets = targets.long().to(self.device)
        # log_probs = F.log_softmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)
        if np.isnan(probs.mean().item()):
            raise NameError('GCE_p')
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze()
        if np.isnan(target_probs.mean().item()):
            raise NameError('GCE_Yg')

        loss_weight = ((1 - target_probs.detach() ** self.q) / self.q)
        loss = F.cross_entropy(logits, targets, reduction='none') * loss_weight
        loss = loss.mean() if ws is None else (loss * ws).mean()
        return loss


def np2ts(ndarray):
    if type(ndarray) == type(np.array([])):
        return torch.tensor(ndarray, dtype=torch.float32)
        # return torch.from_numpy(ndarray).type(torch.float)
    return ndarray


class MLP_plus(nn.Module):
    def __init__(self, input_dim, hidden_dim=[100, 100, 32], output_dim=2, output_dim_distri=2,
                 batch_size=256, epoch=2000, dropout=0, loss_func=None,
                 distri_continual=False,
                 init_mask=0.,
                 optim='Adam', optim_lr=1e-3, device=None, ):
        super(MLP_plus, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') if device is None else device

        self.next_batch_activate_times = 0
        self.next_batch_total_times = None
        self.next_batch_index_array = None
        self.batch_size = batch_size
        self.epoch = epoch
        self.optim_lr = optim_lr
        self.dropout = dropout
        self.output_dim = output_dim
        self.output_dim_distri = output_dim_distri
        self.X_con = None

        self.pruning = False
        self.freeze = False
        self.loss_func = loss_func
        # self.loss_func = GeneralizedCELoss() if loss_func == 'bias' else nn.CrossEntropyLoss()
        self.con_criterion = DebiasedSupConLoss()
        self.distri_continual = distri_continual

        hidden_dim = hidden_dim if type(hidden_dim) == type([]) else [hidden_dim]
        past_dim = input_dim
        self.linear_layer_array = []
        for i_dim in hidden_dim:
            temp_layer = None

            try:
                if self.temp_layer1 is not None:                    pass
                try:
                    if self.temp_layer2 is not None:                        pass
                    # temp_layer = self.temp_layer3 = nn.Linear(past_dim, i_dim)
                    temp_layer = self.temp_layer3 = GateMLP(past_dim, i_dim, init_mask=init_mask).to(self.device)

                except:
                    # temp_layer = self.temp_layer2 = nn.Linear(past_dim, i_dim)
                    temp_layer = self.temp_layer2 = GateMLP(past_dim, i_dim, init_mask=init_mask).to(self.device)
            except:
                # temp_layer = self.temp_layer1 = nn.Linear(past_dim, i_dim)
                temp_layer = self.temp_layer1 = GateMLP(past_dim, i_dim, init_mask=init_mask).to(self.device)

            self.linear_layer_array.append(temp_layer)
            past_dim = i_dim
        self.relu_layer = nn.ReLU()
        # if self.dropout > 0: self.dropout_layer = nn.Dropout(p=dropout)

        self.output_linear_layer = nn.Linear(past_dim, output_dim).to(self.device)
        # self.output_linear_layer = GateMLP(past_dim, output_dim)
        if self.distri_continual:
            self.output_linear_layer_distri = nn.Linear(past_dim, 1).to(self.device)
        else:
            self.output_linear_layer_distri = nn.Linear(past_dim, output_dim_distri).to(self.device)

        self.softmax_layer = nn.Softmax(dim=1)

    def forward(self, X, need_feature=False, need_flip=False):
        if need_flip:
            fliped_X = np2ts(X).to(self.device)
            for i_layer in self.linear_layer_array:
                fliped_X = i_layer(fliped_X, self.pruning, self.freeze, flip=True)
                fliped_X = self.relu_layer(fliped_X)
                # if self.dropout > 0: X = self.dropout_layer(X)
            feature_ = torch.flatten(fliped_X, 1)
            logit = self.output_linear_layer_distri(feature_)
            if need_feature:
                return logit, feature_
            else:
                return logit
        else:
            X = np2ts(X).to(self.device)
            for i_layer in self.linear_layer_array:
                X = i_layer(X, self.pruning, self.freeze)
                X = self.relu_layer(X)
                # if self.dropout > 0: X = self.dropout_layer(X)
            feature_ = torch.flatten(X, 1)
            logit = self.output_linear_layer(feature_)
            if need_feature:
                return logit, feature_
            else:
                return logit

    def fit(self, features, labels, weights=None, distri_flag=False):
        self.train()
        weights = np.ones(len(labels)) if weights is None else weights

        optimizer = optim.Adam(self.parameters(), lr=self.optim_lr)
        features = torch.tensor(features, dtype=torch.float32).to(self.device)
        labels = torch.tensor(labels, dtype=torch.float32).to(self.device)
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

        self.num_samples = len(features)
        self.old_num_samples = self.num_samples

        for epoch in tqdm(range(self.epoch)):
            batch_index = self.next_batches()
            xs, ys, ws = features[batch_index], labels[batch_index], weights[batch_index]

            outputs = self.forward(xs, need_flip=distri_flag)
            # outputs = self.softmax_layer(outputs)

            if self.output_dim != 1:
                if self.loss_func is None:
                    criterion = WeightedCrossEntropyLoss()
                elif 'gradual_bias' in self.loss_func:
                    criterion1 = WeightedCrossEntropyLoss()
                    criterion2 = GeneralizedCELoss()
                elif 'bias' in self.loss_func:
                    criterion = GeneralizedCELoss()
                else:
                    criterion = WeightedCrossEntropyLoss()
                if self.loss_func is None:
                    loss = criterion(outputs.view(-1, self.output_dim), ys, ws)
                elif 'gradual_bias' in self.loss_func:
                    Upper_bound = self.loss_func.split(' ')[1]
                    if '.' in Upper_bound:
                        Upper_bound = self.epoch * float(Upper_bound)
                    else:
                        Upper_bound = int(Upper_bound)

                    if epoch <= Upper_bound:
                        loss1 = criterion1(outputs.view(-1, self.output_dim), ys, ws)
                        loss = loss1
                    else:
                        loss2 = criterion2(outputs.view(-1, self.output_dim), ys, ws)
                        loss = loss2
                else:
                    loss = criterion(outputs.view(-1, self.output_dim), ys, ws)
            else:
                # Temporal loss
                delta = (outputs.view(-1, ) - ys.long().to(self.device))
                loss_main_2 = delta * delta * ws
                loss = loss_main_2.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % (self.epoch // 5) == 0:
                print(f'Epoch [{epoch + 1}/{self.epoch}] - Loss: {loss.item()}')

        index = np.arange(self.num_samples)
        np.random.shuffle(index)
        self.X_con = features[index[:len(features) // 100]]

    def next_batches(self):
        if self.batch_size >= self.num_samples:
            return np.arange(self.num_samples)
        if self.next_batch_total_times is None:
            self.next_batch_total_times = self.num_samples // self.batch_size

        if self.next_batch_index_array is None or self.next_batch_activate_times >= self.next_batch_total_times:
            self.next_batch_index_array = np.arange(self.num_samples)
            np.random.shuffle(self.next_batch_index_array)
            self.next_batch_activate_times = 0

        self.next_batch_activate_times += 1
        cnt = self.next_batch_activate_times
        st, ed = (cnt - 1) * self.batch_size, cnt * self.batch_size
        return_array = self.next_batch_index_array[st: ed]

        return return_array

    def predict_proba(self, input_X, distri_flag=False):
        self.eval()

        input_X = np2ts(input_X).to(self.device)
        with torch.no_grad():
            prob = self.forward(input_X, need_flip=distri_flag)
        # self.train()
        return prob.cpu().numpy()

    def predict(self, input_X, distri_flag=False):
        prob = self.predict_proba(input_X, distri_flag=distri_flag)
        if self.output_dim == 1:
            return np.array(prob).reshape(-1)
        return np.argmax(prob, axis=1)

    def sparsity_regularizer(self, token='gumbel_pi'):
        reg = 0.
        for n, p in self.named_parameters():
            if token in n:
                reg = reg + p.sum()
        return reg

    def pruning_switch(self, turn_on=False):
        self.pruning = turn_on

    def freeze_switch(self, turn_on=False):
        self.freeze = turn_on

    def loss_func_switch(self, loss_func='bias'):
        # self.loss_func = loss_func
        self.loss_func = GeneralizedCELoss() if loss_func == 'bias' else nn.CrossEntropyLoss()

    def count_parameters(self):
        total_params = 0
        effective_params = 0
        for name, module in self.named_modules():
            if isinstance(module, GateMLP):
                total_params += module.weight.numel()
                mask = module.mask.fix_mask_after_pruning()
                effective_params += int(mask.sum().item())
        return total_params, effective_params


if __name__ == "__main__":
    i_path = 'F:/Dataset/Kyoto-2006+/exp/2006_Scaled'
    data = load_cached_data('../MY_general_tools/##tempXy_cache/' + i_path.split('/')[-1] + '.p')
    tempX, tempy = data['tempX'], data['tempy']
    # source = pd.read_csv('F:/Dataset/Kyoto-2006+/exp/2006_Scaled.csv', index_col=None)
    # source = source.sample(50000).reset_index(drop=True)
    # width = source.shape[1]
    # X, y = source.iloc[:, :width - 1].values, source.iloc[:, width - 1].values

    model = MLP(input_dim=tempX.shape[1], num_hidden=64)
    model.fit(tempX, tempy)
    predict = model.predict(tempX)
    test_evaluate(predict, tempy)
