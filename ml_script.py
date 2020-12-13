from utils import *
from models import *
from torch.utils.data import DataLoader
import os
import json
import time
import torch
import torch.optim as optim


def train(lr=0.01, batch_size=4, num_epochs=30, k_fold=1, type_of_image='binary'):
    model_path = 'bs{}_lr{}_epoch{}.pth'.format(batch_size, lr, num_epochs)

    logs = []
    for k in range(k_fold):
        print('Starting CV {}/{}:'.format(k + 1, k_fold))
        net = Net()  # Net() or NewNet()
        log = {}
        log['parameters'] = {'lr': lr, 'batch_size': batch_size, 'num_epochs': num_epochs}
        log['train_loss'] = []
        log['train_acc'] = []
        log['valid_loss'] = []
        log['valid_acc'] = []

        # Hyperparameters
        optimizer = optim.SGD(net.parameters(), lr=lr)  # or optim.SGD/Adam

        train_dataset, valid_dataset, test_dataset = prepare_datasets(my_seed=k, type_of_image=type_of_image)

        trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
        validloader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=2,
                                 drop_last=True)  # no need to shuffle - we're just evaluating the accuracy
        testloader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2,
                                drop_last=True)  # no need to shuffle - we're just evaluating the accuracy

        criterion = nn.CrossEntropyLoss()

        start = time.time()
        for epoch in range(num_epochs):  # loop over the dataset multiple times

            running_train_loss = 0.0
            total_correct = 0
            total_images = 0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                images, labels = data['image'], data['label']

                # zero the parameter gradients (otherwise they retain gradients from the previous iteration)
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(images)
                loss = criterion(outputs, labels)
                running_train_loss += loss.item()

                # let's also get our training accuracy too (for the sake of being complete)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                total_correct += sum(c).item()
                total_images += images.shape[0]

                loss.backward()  # backpropagation
                optimizer.step()  # GD, SGD, whichever one you choose in the previous section

            cur_train_acc = total_correct / total_images

            # what if we get our own

            # Often times validation is performed at the end of each epoch
            total_correct = 0
            total_images = 0
            best_val_acc = 0
            running_valid_loss = 0
            with torch.no_grad():
                for data in validloader:
                    images, labels = data['image'], data['label']  # labels are still 0-indexed
                    outputs = net(images)
                    loss = criterion(outputs, labels)
                    running_valid_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    c = (predicted == labels).squeeze()
                    total_correct += sum(c).item()
                    total_images += images.shape[0]
                cur_val_acc = total_correct / total_images
                print('Epoch %d, train_loss: %.3f, train_acc: %.3f, val_loss: %.3f, val_acc = %.2f' % (epoch + 1,
                                                                                                       running_train_loss / len(
                                                                                                           trainloader),
                                                                                                       cur_train_acc,
                                                                                                       running_valid_loss / len(
                                                                                                           validloader),
                                                                                                       cur_val_acc))
                # record metrics
                log['train_loss'].append(running_train_loss / len(trainloader))
                log['train_acc'].append(cur_train_acc)
                log['valid_loss'].append(running_valid_loss / len(validloader))
                log['valid_acc'].append(cur_val_acc)

                if cur_val_acc > best_val_acc:
                    best_val_acc = cur_val_acc
                    torch.save(net.state_dict(), model_path)

        end = time.time()
        log['computation_time (min)'] = (end - start) / 60
        log['model_path'] = model_path

        # test the best performing epoch
        net = Net()
        net.load_state_dict(torch.load(model_path))
        total_correct = 0
        total_images = 0
        class_correct = np.zeros(3)
        class_total = np.zeros(3)

        with torch.no_grad():
            for data in testloader:
                images, labels = data['image'], data['label']
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                total_correct += sum(c).item()
                total_images += images.shape[0]
                for i in range(4):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
            log['test_overall_acc'] = total_correct / total_images
            log['test_class_acc'] = list(class_correct / class_total)

        logs.append(log)

        if k_fold == 1:
            logs = log

    return logs
