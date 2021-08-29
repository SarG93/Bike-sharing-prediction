import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader


def train(args, model: nn.Module, dataloader: DataLoader):
    optimizer = optim.Adam(list(model.parameters()), lr=args.learning_rate)
    model.train()
    l1_loss = nn.L1Loss(reduction='none')
    best_loss = np.inf
    num_batches = len(dataloader.dataset)/args.batch_size
    for epoch in range(args.num_epochs):

        epoch_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            model.zero_grad()

            output = model(batch['input'].float())
            loss = l1_loss(output, batch['labels'].float())
            mae = torch.sum(torch.mean(loss, dim=0))
            mae.backward()
            optimizer.step()
            epoch_loss += mae

        print(f'epoch: {epoch}/{args.num_epochs}, loss: {epoch_loss / num_batches}')
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), os.path.join('./models', 'bike_count_mlp.dat'))


def evaluate(args, model: nn.Module, dataloader: DataLoader):
    model.eval()
    mean_expected_error = 0
    l1_loss = nn.L1Loss(reduction='none')
    labels = []
    predictions = []
    num_batches = len(dataloader.dataset) / args.batch_size
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            model.zero_grad()
            output = model(batch['input'].float())
            loss = l1_loss(output, batch['labels'].float())
            mae = torch.sum(torch.mean(loss, dim=0))
            mean_expected_error += mae
            labels += list(torch.sum(batch['labels'], dim=1).numpy())
            predictions += list(np.around(torch.sum(output, dim=1).numpy()))

    print(f'Mean Expected Error: {mean_expected_error / num_batches}')
    plt.scatter(labels, predictions, alpha=0.4)
    plt.plot(range(1000), range(1000), 'r')
    plt.xlabel('True count')
    plt.ylabel('Predicted count')
    plt.title('Multi Layer Perceptron')
    plt.savefig(os.path.join('./plots', 'scatter_plot_MLP.png'))
    plt.close()
    plt.plot(list(range(200)), labels[:200], label='true')
    plt.plot(list(range(200)), predictions[:200], label='predicted')
    plt.ylabel('Bike counts')
    plt.xlabel('Number of hours')
    plt.legend()
    plt.savefig(os.path.join('./plots', 'prediction_200hrs.png'))
