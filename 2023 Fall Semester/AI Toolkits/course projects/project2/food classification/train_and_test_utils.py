import torch
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(model, optimizer, sample):
    model.train()

    criterion = nn.CrossEntropyLoss()

    optimizer.zero_grad()

    img = sample['img'].to(device)
    label = sample['id'].to(device)

    pred = model(img)

    num_correct = torch.sum(torch.argmax(pred, dim=-1) == label)

    pred_loss = criterion(pred, label)

    pred_loss.backward()

    optimizer.step()

    return pred_loss.item(), num_correct.item()


def test(model, sample):
    model.eval()

    with torch.no_grad():

        criterion = nn.CrossEntropyLoss()

        img = sample['img'].to(device)
        label = sample['id'].to(device)
        pred = model(img)
        num_correct = torch.sum(torch.argmax(pred, dim=-1) == label)
        pred_loss = criterion(pred, label)

    return pred_loss.item(), num_correct.item()
