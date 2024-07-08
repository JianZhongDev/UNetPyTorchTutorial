"""
FILENAME: TrainTest.py
DESCRIPTION: Training and testing functions
@author: Jian Zhong
"""

import torch


# train model for one epoch
def train_one_epoch(model, train_loader, loss_func, optimizer, device):
    tot_loss = 0.0
    avg_loss = 0.0
    tot_nof_batch = 0

    model.to(device)
    model.train(True)
    for i_batch, data in enumerate(train_loader):

        inputs = None
        targets = None
        weights = None

        inputs = data[0]
        targets = data[1]
        if len(data) > 2:
            weights = data[2]

        inputs = inputs.to(device)
        targets = targets.to(device)
        if weights is not None:
            weights = weights.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)

        if weights is None:
            loss = loss_func(outputs, targets)
        else:
            loss = loss_func(outputs, targets, weights)


        loss.backward()
        optimizer.step()

        tot_loss += loss.item()
        tot_nof_batch += 1

        if i_batch % 10 == 0:
            print(f"batch {i_batch} loss: {tot_loss/tot_nof_batch : >8f}")

    avg_loss = tot_loss/tot_nof_batch

    print(f"Train: Avg loss: {avg_loss:>8f}")

    return avg_loss


#  validate model for one epoch
def validate_one_epoch(model, validate_loader, loss_func, device):
    tot_loss = 0.0
    avg_loss = 0.0
    tot_nof_batch = len(validate_loader)

    model.to(device)
    model.eval()
    with torch.no_grad():
        for i_batch, data in enumerate(validate_loader):

            inputs = None
            targets = None
            weights = None

            inputs = data[0]
            targets = data[1]
            if len(data) > 2:
                weights = data[2]
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            if weights is not None:
                weights = weights.to(device)
            
            outputs = model(inputs)

            if weights is None:
                loss = loss_func(outputs, targets)
            else:
                loss = loss_func(outputs, targets, weights)

            tot_loss += loss.item()

    avg_loss = tot_loss/tot_nof_batch

    print(f"Validate: Avg loss: {avg_loss:>8f}")

    return avg_loss