import sys
import timeit

import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from main.package import send_to_wechat
import preprocess as pp
import wandb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
matplotlib.use('Agg')
from package import MolecularGraphNeuralNetwork, Tester, Trainer, plot_fit_confidence_bond


if __name__ == "__main__":
    project = 'smtr'
    wandb.login(key='local-8fe6e6b5840c4c05aaaf6aac5ca8c1fb58abbd1f', host='http://localhost:8080')

    wandb.init(project=project)
    dataset = 'smtr'
    # dataset=yourdataset

    # The molecular property to be learned.
    property = 'e(kcalmol^-1)'
    # property='HOMO(eV)'
    # property='LUMO(eV)'

    # The setting of a neural network architecture.
    dim = 100
    layer_hidden = 24
    layer_output = 24

    # The setting for optimization.
    batch_train = 128
    batch_test = 32
    lr = 1e-4
    lr_decay = 0.997
    decay_interval = 10
    iteration = 3000
    wandb.config.update({
        "learning_rate": lr,
        "epochs": iteration,
        "batch_size": batch_train,
        'dim': dim,
        'layer_hidden': layer_hidden,
        'layer_output': layer_output,
        'lr_decay': lr_decay,
        'decay_interval': decay_interval
    })
    run_timer = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
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
    (dataset_train, idx_train, dataset_val, idx_val, dataset_test, idx_test,
     N_atoms) = pp.create_datasets(dataset, property, device)
    print('-'*100)
    setting = f'test_{batch_train}_{lr}_{N_atoms}_{dim}_{layer_hidden}_{layer_output}_{run_timer}'
    print('The preprocess has finished!')
    print('# of training data samples:', len(dataset_train))
    print('# of development data samples:', len(dataset_val))
    print('# of test data samples:', len(dataset_test))
    print('-'*100)

    print('Creating a model.')
    torch.manual_seed(1234)  # initialize the model with a random seed.
    model = MolecularGraphNeuralNetwork(
            N_atoms, dim, layer_hidden, layer_output, device=device).to(device)
    trainer = Trainer(model, lr=lr, batch_train=batch_train)
    tester = Tester(model, batch_test=batch_test)
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
    ts = np.array(list())
    ys = np.array(list())
    for epoch in range(iteration):

        epoch += 1
        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train = trainer.train(dataset_train)
        error_train = tester.test(dataset_train)
        error_val = tester.test(dataset_val)
        error_test = tester.test(dataset_test)

        time_tmp = timeit.default_timer() - start
        wandb.log({"loss": loss_train, 'error_val': error_val, 'error_train': error_train})

        # Optional
        # wandb.watch(model)
        if epoch == 1:
            minutes = time_tmp * iteration / 60
            hours = int(minutes / 60)
            minutes = int(minutes - 60 * hours)
            print('The training will finish in about',
                  hours, 'hours', minutes, 'minutes.')
            print('-'*100)
            print(result)

        result = '\t'.join(map(str, [epoch, time_tmp, loss_train,
                                     error_val, error_test]))
        tester.save_result(result, file_result)

        print(result)
    # wandb.log({"test_ts": test_ts, 'test_ys': test_ys})
    torch.save(model.state_dict(), f'../model/{setting}.params')
    print('The training has finished!')

    val_batch = list(zip(*dataset_val))
    train_batch = list(zip(*dataset_train[:160]))
    test_batch = list(zip(*dataset_test))
    _, ts_val, ys_val = model(val_batch, train=False)
    _, ts_train, ys_train = model(train_batch, train=False)
    _, ts_test, ys_test = model(test_batch, train=False)

    r2_val = r2_score(ts_val, ys_val)
    r2_train = r2_score(ts_train, ys_train)
    r2_test = r2_score(ts_test, ys_test)
    fig = plot_fit_confidence_bond(ts_val, ys_val, r2_val)
    wandb.log({'chart': fig})
    wandb.finish()
    message2wechat = {'title': project+' Train Results'}
    message2wechat['desp'] = f'r2_val: {r2_val}\nr2_train: {r2_train}\nr2_test: {r2_test}'
    alw_2wechat = True
    if int(time.strftime("%H", time.localtime())) in list(range(0, 8))+[23, 24] or alw_2wechat:
        send2wechat = True
    if send2wechat:
        send_to_wechat(message2wechat)

