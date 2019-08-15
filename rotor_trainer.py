import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import math
from torch.utils.data.sampler import SubsetRandomSampler

# Custom classes
from label_generator import LabelGenerator
from rotor_dataset import RotorDataset
from simple_net import SimpleNet
from train import train


def main():

    # Initialize parameters
    num_epochs = 10
    validation_split = .1
    batch_size = 20
    steering_label_names = ['left', 'half_left', 'neutral', 'half_right', 'right']
    steering_value_range = (-1, 1)

    # Create labels
    print("Creating label generator...")
    steering_label_generator = LabelGenerator(steering_label_names, steering_value_range)

    # Get working directories
    current_working_dir = os.getcwd()
    data_dir = os.path.join(current_working_dir, 'data/gen_track_user_drv_right_lane')

    # Initialize dataset
    print("Creating dataset...")
    dataset = RotorDataset(data_dir, steering_label_generator)

    # Create model
    print("Creating model...")
    model = SimpleNet()

    # Initialize optimizer
    print("Initializing optimizer...")
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # Create training and validation loaders
    data_size = len(dataset)
    split_ind = math.floor((1 - validation_split) * data_size)
    data_indices = list(range(data_size))
    train_ind, validation_ind = data_indices[0:split_ind], data_indices[split_ind:]

    train_sampler = SubsetRandomSampler(train_ind)
    validation_sampler = SubsetRandomSampler(validation_ind)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=validation_sampler)

    # Train the model
    print("Training...")
    loss_list = []
    accuracy_list = []
    for epoch in range(0, num_epochs):
        model, loss, accuracy = train(epoch, model, optimizer, train_loader, validation_loader)
        loss_list.extend(loss)
        accuracy_list.append(accuracy)
        

if __name__ == '__main__':

    main()
