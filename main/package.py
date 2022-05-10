
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import requests

class MolecularGraphNeuralNetwork(nn.Module):
    def __init__(self, N_atoms, dim, layer_hidden, layer_output, device):
        super(MolecularGraphNeuralNetwork, self).__init__()
        self.embed_atom = nn.Embedding(N_atoms, dim)
        self.gamma = nn.ModuleList([nn.Embedding(N_atoms, 1)
                                    for _ in range(layer_hidden)])
        self.W_atom = nn.ModuleList([nn.Linear(dim, dim)
                                     for _ in range(layer_hidden)])
        self.W_output = nn.ModuleList([nn.Linear(dim, dim)
                                       for _ in range(layer_output)])
        self.W_property = nn.Linear(dim, 1)
        self.device = device
        self.N_atoms = N_atoms
        self.dim = dim
        self.layer_hidden = layer_hidden
        self.layer_output = layer_output
    def pad(self, matrices, pad_value):
        """Pad the list of matrices
        with a pad_value (e.g., 0) for batch processing.
        For example, given a list of matrices [A, B, C],
        we obtain a new matrix [A00, 0B0, 00C],
        where 0 is the zero (i.e., pad value) matrix.
        """
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        zeros = torch.FloatTensor(np.zeros((M, N))).to(self.device)
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i:i+m, j:j+n] = matrix
            i += m
            j += n
        return pad_matrices

    def update(self, matrix, vectors, layer):
        hidden_vectors = torch.relu(self.W_atom[layer](vectors))
        return hidden_vectors + torch.matmul(matrix, hidden_vectors)

    def sum(self, vectors, axis):
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(sum_vectors)

    def forward(self, inputs):

        """Cat or pad each input data for batch processing."""
        atoms, distance_matrices, molecular_sizes = inputs
        atoms = torch.cat(atoms)
        distance_matrix = self.pad(distance_matrices, 1e6)

        """GNN layer (update the atom vectors)."""
        atom_vectors = self.embed_atom(atoms)
        for l in range(self.layer_hidden):
            gammas = torch.squeeze(self.gamma[l](atoms))
            M = torch.exp(-gammas*distance_matrix**2)
            atom_vectors = self.update(M, atom_vectors, l)
            atom_vectors = F.normalize(atom_vectors, 2, 1)  # normalize.

        """Output layer."""
        for l in range(self.layer_output):
            atom_vectors = torch.relu(self.W_output[l](atom_vectors))

        """Molecular vector by sum of the atom vectors."""
        molecular_vectors = self.sum(atom_vectors, molecular_sizes)

        """Molecular property."""
        properties = self.W_property(molecular_vectors)

        return properties

    def __call__(self, data_batch, train):

        inputs = data_batch[:-1]
        correct_properties = torch.cat(data_batch[-1])

        if train:
            predicted_properties = self.forward(inputs)
            loss = F.mse_loss(predicted_properties, correct_properties)
            return loss
        else:
            with torch.no_grad():
                predicted_properties = self.forward(inputs)
            ts = correct_properties.to('cpu').data.numpy()
            ys = predicted_properties.to('cpu').data.numpy()
            ts, ys = np.concatenate(ts), np.concatenate(ys)
            sum_absolute_error = sum(np.abs(ts-ys))
            return sum_absolute_error, ts, ys


class Trainer(object):
    def __init__(self, model, lr, batch_train):
        self.model = model
        self.lr = lr
        self.batch_train = batch_train
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, dataset):
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        for i in range(0, N, self.batch_train):
            data_batch = list(zip(*dataset[i:i+self.batch_train]))
            loss = self.model(data_batch, train=True)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.item()
        return loss_total


class Tester(object):
    def __init__(self, model, batch_test):
        self.model = model
        self.batch_test = batch_test
    def test(self, dataset):
        N = len(dataset)
        SAE = 0
        for i in range(0, N, self.batch_test):
            data_batch = list(zip(*dataset[i:i+self.batch_test]))
            sum_absolute_error, _, _ = self.model(data_batch, train=False)
            SAE += sum_absolute_error
        MAE = SAE / N

        return MAE

    def save_result(self, result, filename):
        with open(filename, 'a') as f:
            f.write(result + '\n')


def plot_fit_confidence_bond(x, y, r2):
    # fit a linear curve an estimate its y-values and their error.
    a, b = np.polyfit(x, y, deg=1)
    y_est = a * x + b
    # y_err = x.std() * np.sqrt(1 / len(x) +
    #                           (x - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))

    fig, ax = plt.subplots()
    ax.plot([-20, 0], [-20, 0], '-')
    ax.plot(x, y_est, '-')
    # ax.fill_between(x, y_est - y_err, y_est + y_err, alpha=0.2)
    ax.plot(x, y, 'o', color='tab:brown')
    num = 0
    for x_i, y_i in zip(x, y):
        ax.annotate(str(num), (x_i, y_i))
        num += 1
    ax.set_xlabel('True Energy(Kcal/mol)')
    ax.set_ylabel('Predict Energy(Kcal/mol)')
    # ax.text(0.1, 0.5, 'r2:  ' + str(r2))
    ax.text(0.4, 0.9,
            'r2:  ' + str(r2), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
            fontsize=12)
    return fig


def send_to_wechat(message):
    key = 'SCT67936Tpp9RtEM5SnSNxczhMTKaMzW1'
    url = f'https://sctapi.ftqq.com/{key}.send'
    return requests.post(url=url, data=message)
