import sys
import timeit

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import preprocess as pp
import wandb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score


class MolecularGraphNeuralNetwork(nn.Module):
    def __init__(self, N_atoms, dim, layer_hidden, layer_output):
        super(MolecularGraphNeuralNetwork, self).__init__()
        self.embed_atom = nn.Embedding(N_atoms, dim)
        self.gamma = nn.ModuleList([nn.Embedding(N_atoms, 1)
                                    for _ in range(layer_hidden)])
        self.W_atom = nn.ModuleList([nn.Linear(dim, dim)
                                     for _ in range(layer_hidden)])
        self.W_output = nn.ModuleList([nn.Linear(dim, dim)
                                       for _ in range(layer_output)])
        self.W_property = nn.Linear(dim, 1)

    def pad(self, matrices, pad_value):
        """Pad the list of matrices
        with a pad_value (e.g., 0) for batch processing.
        For example, given a list of matrices [A, B, C],
        we obtain a new matrix [A00, 0B0, 00C],
        where 0 is the zero (i.e., pad value) matrix.
        """
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        zeros = torch.FloatTensor(np.zeros((M, N))).to(device)
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
        for l in range(layer_hidden):
            gammas = torch.squeeze(self.gamma[l](atoms))
            M = torch.exp(-gammas*distance_matrix**2)
            atom_vectors = self.update(M, atom_vectors, l)
            atom_vectors = F.normalize(atom_vectors, 2, 1)  # normalize.

        """Output layer."""
        for l in range(layer_output):
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
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, dataset):
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        for i in range(0, N, batch_train):
            data_batch = list(zip(*dataset[i:i+batch_train]))
            loss = self.model(data_batch, train=True)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.item()
        return loss_total


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset):
        N = len(dataset)
        SAE = 0
        ts = list()
        ys = list()
        for i in range(0, N, batch_test):
            data_batch = list(zip(*dataset[i:i+batch_test]))
            sum_absolute_error, ts_t, ys_t = self.model(data_batch, train=False)
            SAE += sum_absolute_error
        MAE = SAE / N
        ts = np.append(ts, ts_t)
        ys = np.append(ys, ys_t)

        return MAE, ts, ys

    def save_result(self, result, filename):
        with open(filename, 'a') as f:
            f.write(result + '\n')


def plot_fit_confidence_bond(x, y, r2):
    # fit a linear curve an estimate its y-values and their error.
    a, b = np.polyfit(x, y, deg=1)
    y_est = a * x + b
    y_err = x.std() * np.sqrt(1 / len(x) +
                              (x - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))

    fig, ax = plt.subplots()
    ax.plot(x, y_est, '-')
    # ax.fill_between(x, y_est - y_err, y_est + y_err, alpha=0.2)
    ax.plot(x, y, 'o', color='tab:brown')
    # ax.text(0.1, 0.5, 'r2:  ' + str(r2))
    ax.text(0.1, 0.9, 'r2:  ' + str(r2), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    wandb.log({'chart': fig})

if __name__ == "__main__":
    wandb.login(key='local-8fe6e6b5840c4c05aaaf6aac5ca8c1fb58abbd1f', host='http://localhost:8080')
    wandb.init('smtr')
    dataset = 'smtr'
    # dataset=yourdataset

    # The molecular property to be learned.
    property = 'e(kcalmol^-1)'
    # property='HOMO(eV)'
    # property='LUMO(eV)'

    # The setting of a neural network architecture.
    dim = 50
    layer_hidden = 6
    layer_output = 6

    # The setting for optimization.
    batch_train = 128
    batch_test = 32
    lr = 1e-5
    lr_decay = 0.99
    decay_interval = 10
    iteration = 3000
    wandb.config = {
        "learning_rate": lr,
        "epochs": iteration,
        "batch_size": batch_train
    }
    setting = f'test_{batch_train}_{lr}'
    # (dataset, property, dim, layer_hidden, layer_output,
    #  batch_train, batch_test, lr, lr_decay, decay_interval, iteration,
    #  setting) = sys.argv[1:]
    (dim, layer_hidden, layer_output, batch_train, batch_test, decay_interval,
     iteration) = map(int, [dim, layer_hidden, layer_output, batch_train,
                            batch_test, decay_interval, iteration])
    lr, lr_decay = map(float, [lr, lr_decay])

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses a GPU!')
    else:
        device = torch.device('cpu')
        print('The code uses a CPU...')
    print('-'*100)

    print('Preprocessing the', dataset, 'dataset.')
    print('Just a moment......')
    (dataset_train, dataset_dev, dataset_test,
     N_atoms) = pp.create_datasets(dataset, property, device)
    print('-'*100)

    print('The preprocess has finished!')
    print('# of training data samples:', len(dataset_train))
    print('# of development data samples:', len(dataset_dev))
    print('# of test data samples:', len(dataset_test))
    print('-'*100)

    print('Creating a model.')
    torch.manual_seed(1234)  # initialize the model with a random seed.
    model = MolecularGraphNeuralNetwork(
            N_atoms, dim, layer_hidden, layer_output).to(device)
    trainer = Trainer(model)
    tester = Tester(model)
    print('# of model parameters:',
          sum([np.prod(p.size()) for p in model.parameters()]))
    print('-'*100)

    for i in range(layer_hidden):
        ones = nn.Parameter(torch.ones((N_atoms, 1))).to(device)
        model.gamma[i].weight.data = ones  # initialize gamma with ones.

    file_result = '../output/result--' + setting + '.txt'

    result = ('Epoch\tTime(sec)\tLoss_train(MSE)\t'
              'Error_dev(MAE)\tError_test(MAE)')
    with open(file_result, 'w') as f:
        f.write(result + '\n')

    print('Start training.')
    print('The result is saved in the output directory every epoch!')

    np.random.seed(1234)

    start = timeit.default_timer()

    for epoch in range(iteration):

        epoch += 1
        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train = trainer.train(dataset_train)
        error_dev, dev_ts, dev_ys = tester.test(dataset_dev)
        error_test, test_ts, test_ys = tester.test(dataset_test)

        time = timeit.default_timer() - start
        wandb.log({"loss": loss_train, 'error_test': error_test})

        # Optional
        # wandb.watch(model)
        if epoch == 1:
            minutes = time * iteration / 60
            hours = int(minutes / 60)
            minutes = int(minutes - 60 * hours)
            print('The training will finish in about',
                  hours, 'hours', minutes, 'minutes.')
            print('-'*100)
            print(result)

        result = '\t'.join(map(str, [epoch, time, loss_train,
                                     error_dev, error_test]))
        tester.save_result(result, file_result)

        print(result)
    # wandb.log({"test_ts": test_ts, 'test_ys': test_ys})
    print('The training has finished!')
    r2 = r2_score(test_ts, test_ys)
    plot_fit_confidence_bond(test_ts, test_ys, r2)
    wandb.finish()
