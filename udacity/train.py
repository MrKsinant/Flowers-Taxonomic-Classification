#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#########################################################################
# PROGRAMMER: Pierre-Antoine Ksinant                                    #
# DATE CREATED: 15/07/2018                                              #
# REVISED DATE: -                                                       #
# PURPOSE: Train a chosen neural network on a dataset to predict flower #
#          name from an image along with the probability of that name   #
#                                                                       #
# Expected call with <> indicating expected user input:                 #
#      python train.py <data/directory> --save_dir <save_directory>     #
#             --arch <arch> --learning_rate <learning_rate>             #
#             --hidden_units <hidden_units> --epochs <epochs> --gpu     #
#                                                                       #
# Example call:                                                         #
#      python train.py ../aipnd/flowers --save_dir checkpoints_models   #
#             --arch densenet161 --learning_rate 0.0001                 #
#             --hidden_units 500 --epochs 2 --gpu                       #
#########################################################################

###########################
# Needed packages imports #
###########################

import argparse, os, torch, time, sys
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
from workspace import active_session

#########################
# Main program function #
#########################

def main():

    # Creates 7 command line arguments:

    parser = argparse.ArgumentParser(description='Train a chosen neural network on a dataset to predict flower name from an image along with the probability of that name.')

    parser.add_argument('data_directory', type=str,
                        help='path to data_directory for training')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='set directory to save checkpoints')
    parser.add_argument('--arch', type=str, choices=['densenet161', 'vgg16'],
                        default='densenet161',
                        help='choose architecture, densenet161 or vgg16 (default densenet161)')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='set learning rate for training (default 0.0001)')
    parser.add_argument('--hidden_units', type=int, default=500,
                        help='set hidden units for training (default 500)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='set epochs for training (default 1)')
    parser.add_argument('--gpu', action='store_true',
                        help='use GPU for training')

    in_arg = vars(parser.parse_args())

    # Checks command line arguments and set variables:

    data_dir = in_arg['data_directory']
    if not(os.path.isdir(data_dir)):
        print("The directory '{}' can't be found.".format(data_dir))
        sys.exit(1)

    save_dir = in_arg['save_dir']
    if save_dir != None and not(os.path.isdir(save_dir)):
        os.mkdir(save_dir)

    arch = in_arg['arch']

    learning_rate = in_arg['learning_rate']

    hidden_units = in_arg['hidden_units']

    epochs = in_arg['epochs']

    use_gpu = in_arg['gpu']
    if use_gpu and not(torch.cuda.is_available()):
        print("GPU mode is not available.")
        sys.exit(1)

    # Shortcut variables to the training, validation and testing datasets:

    training_dir = data_dir + '/train'
    validation_dir = data_dir + '/valid'
    testing_dir = data_dir + '/test'

    # Transforms for the training, validation and testing datasets:

    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

    validate_test_transforms = transforms.Compose([transforms.Resize(256),
                                                  transforms.CenterCrop(224),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406],
                                                                      [0.229, 0.224, 0.225])])

    # Load the training, validation and testing datasets with ImageFolder:

    training_dataset = datasets.ImageFolder(training_dir, transform=train_transforms)
    validation_dataset = datasets.ImageFolder(validation_dir, transform=validate_test_transforms)
    testing_dataset = datasets.ImageFolder(testing_dir, transform=validate_test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders:

    trainingloader = torch.utils.data.DataLoader(training_dataset, batch_size=64, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_dataset, batch_size=32)
    testingloader = torch.utils.data.DataLoader(testing_dataset, batch_size=32)

    # Choose a pretrained neural network:

    if arch == 'densenet161':

        model = models.densenet161(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False

        classifier = nn.Sequential(OrderedDict([('input', nn.Linear(2208, hidden_units)),
                                                ('drop1', nn.Dropout(p=0.2)),
                                                ('act1', nn.ReLU()),
                                                ('hl1', nn.Linear(hidden_units, 102)),
                                                ('output', nn.LogSoftmax(dim=1))]))

        model.classifier = classifier

    else:

        model = models.vgg16(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False

        classifier = nn.Sequential(OrderedDict([('input', nn.Linear(25088, hidden_units)),
                                                ('drop1', nn.Dropout(p=0.2)),
                                                ('act1', nn.ReLU()),
                                                ('hl1', nn.Linear(hidden_units, 102)),
                                                ('output', nn.LogSoftmax(dim=1))]))

        model.classifier = classifier

    # Define criterion and optimizer:

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # Create a performance tracking file for the training session:

    performance_file = open('../graphs/performance-tracking.txt', 'w')
    performance_file.write("# Epoch, Training Loss, Training Accuracy, Validation Loss, Validation Accuracy\n")
    performance_file.close()

    # Training of the chosen pretrained neural network:

    print("*** TRAINING SESSION")

    start_time = time.time()

    if use_gpu:
        model.to('cuda')
    else:
        model.to('cpu')

    # Maintain the session active:
    with active_session():

        for e in range(epochs):
            print("Epoch {}".format(e+1))

            # Forward and backward loop:
            model.train()
            running_loss = 0
            for data in trainingloader:
                inputs, labels = data
                if use_gpu:
                    inputs, labels = inputs.to('cuda'), labels.to('cuda')
                else:
                    inputs, labels = inputs.to('cpu'), labels.to('cpu')
                optimizer.zero_grad()
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # Check loss and accuracy on the training dataset:
            model.eval()
            running_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for data in trainingloader:
                    images, labels = data
                    if use_gpu:
                        images, labels = images.to('cuda'), labels.to('cuda')
                    else:
                        images, labels = images.to('cpu'), labels.to('cpu')
                    outputs = model.forward(images)
                    loss = criterion(outputs, labels)
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print("Training: Loss... {:.2f},".format(running_loss),
                  "Accuracy... {:.2f}%".format(100 * correct / total))

            performance_file = open('../graphs/performance-tracking.txt', 'a')
            performance_file.write("{} {:.2f} {:.2f}".format(e+1, running_loss, 100 * correct / total))

            # Check loss and accuracy on the validation dataset:
            model.eval()
            running_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for data in validationloader:
                    images, labels = data
                    if use_gpu:
                        images, labels = images.to('cuda'), labels.to('cuda')
                    else:
                        images, labels = images.to('cpu'), labels.to('cpu')
                    outputs = model.forward(images)
                    loss = criterion(outputs, labels)
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                print("Validation: Loss... {:.2f},".format(running_loss),
                      "Accuracy... {:.2f}%".format(100 * correct / total))

                performance_file.write(" {:.2f} {:.2f}\n".format(running_loss, 100 * correct / total))
                performance_file.close()

    # Time performance:
    end_time = time.time()
    total_time = int(end_time - start_time)
    hours = total_time // 3600
    minutes = (total_time % 3600) // 60
    seconds = (total_time % 3600) % 60

    print("Time: {:02d}h {:02d}m {:02d}s".format(hours, minutes, seconds))

    # Test the model obtained on the testing dataset:

    print("*** TESTING SESSION")

    model.eval()
    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testingloader:
            images, labels = data
            if use_gpu:
                images, labels = images.to('cuda'), labels.to('cuda')
            else:
                images, labels = images.to('cpu'), labels.to('cpu')
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print("Loss... {:.2f},".format(running_loss),
          "Accuracy... {:.2f}%".format(100 * correct / total))

    # Model's checkpoint:

    if save_dir != None:
        os.chdir(save_dir)

    model.to('cpu')

    checkpoint = {'arch': arch,
                  'learning_rate': learning_rate,
                  'hidden_units': hidden_units,
                  'epochs': epochs,
                  'state_dict': model.state_dict()}

    model_name = 'arch_' + arch + '_lr_' + str(learning_rate) + '_hu_' + str(hidden_units) + '_epochs_' + str(epochs) + '.pth'

    torch.save(checkpoint, model_name)

    # Print summary report:

    print("*** SUMMARY REPORT")
    print("Data directory: {}".format(data_dir))
    print("Save directory: {}".format(save_dir))
    print("Architecture: {}".format(arch))
    print("Learning rate: {}".format(learning_rate))
    print("Hidden units: {}".format(hidden_units))
    print("Epoch(s): {}".format(epochs))
    if use_gpu:
        print("Mode: GPU")
    else:
        print("Mode: CPU")
    print("*** CHECKPOINT MODEL NAME")
    print(model_name)

############################################
# Call to main function to run the program #
############################################

if __name__ == "__main__":
    main()
