import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time, sleep

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn, optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    cuda = True if torch.cuda.is_available() else False
    device = torch.device('cpu')
    if cuda:
        device = torch.device('cuda')

    batch_size = 128
    #Inicijalizacija writera
    writer = SummaryWriter('runs/MNIST')

    transforms = torchvision.transforms.Compose([
        transforms.ToTensor(),
        torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
        torchvision.transforms.RandomRotation(5, interpolation=torchvision.transforms.InterpolationMode.BILINEAR),

        transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = datasets.EMNIST('emnist_data',split='letters', download=True, train=True, transform=transforms)
    testset = datasets.EMNIST('emnist_data',split='letters', download=True, train=False, transform=transforms)


    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)



    input_size = 784
    hidden_size_1 = 128
    hidden_size_2 = 256
    hidden_size_3 = 128
    hidden_size_4 = 64
    output_size = 27


    model = nn.Sequential(nn.Linear(input_size, hidden_size_1),
                          nn.ReLU(),
                          nn.Linear(hidden_size_1, hidden_size_2),
                          nn.ReLU(),
                          nn.Linear(hidden_size_2, hidden_size_3),
                          nn.ReLU(),
                          nn.Linear(hidden_size_3, hidden_size_4),
                          nn.ReLU(),
                          nn.Linear(hidden_size_4, output_size),
                          nn.LogSoftmax(dim=1)).to(device)


    loss_fn = nn.NLLLoss().to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    epochs = 30
    train_per_epoch = int(len(trainset) / batch_size)
    for e in range(epochs):
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
        for idx, (images, labels) in loop:

            images = images.to(device, non_blocking=True).view(images.shape[0], -1)
            optimizer.zero_grad()
            output = model(images)
            labels = labels.to(device, non_blocking=True)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

            writer.add_scalar('loss', loss.item(), (e * train_per_epoch) + idx)
            predictions = output.argmax(dim=1, keepdim=True).squeeze()
            correct = (predictions == labels).sum().item()
            accuracy = correct / len(predictions)
            loop.set_description(f"Epoch [{e}/{epochs}")
            loop.set_postfix(loss=loss.item(), acc=accuracy)
            writer.add_scalar('acc', accuracy, (e * train_per_epoch) + idx)
        # provjera točnosti na novim testnom setu unutar petlje za učenje
        num_correct = 0
        num_samples = 0
        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                img = images.to(device, non_blocking=True).view(images.shape[0], -1)
                labels = labels.to(device=device)

                scores = model(img)
                _, predictions = scores.max(1)
                num_correct += (predictions == labels).sum()
                num_samples += predictions.size(0)

            print(
                f'\nValidacijski set ima točnih {num_correct} od ukupno {num_samples} što čini točnost od {float(num_correct) / float(num_samples) * 100:.2f}%')
    #provjera tocnosti nad testnim setom na kraju
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            img = images.to(device, non_blocking=True).view(images.shape[0], -1)
            labels = labels.to(device=device)

            scores = model(img)
            _, predictions = scores.max(1)
            num_correct += (predictions == labels).sum()
            num_samples += predictions.size(0)

        print(
            f'\nValidacijski set ima točnih {num_correct} od ukupno {num_samples} što čini točnost od {float(num_correct) / float(num_samples) * 100:.2f}%')