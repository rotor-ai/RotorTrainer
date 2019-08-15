# Import data sets
import pathlib
import os

current_working_dir = os.getcwd()
data_dir = os.path.join(current_working_dir, 'data/donkey_car_example/gen_track_user_drv_right_lane')

def value_to_label(val):
    if val < -0.5:
        return 'left'
    elif val < 0:
        return 'half_left'
    elif val == 0:
        return 'neutral'
    elif val < 0.5:
        return 'half_right'
    else:
        return 'right'
    
def value_to_label_index(value):
    return label_to_index[value_to_label(value)]

label_names = ['left', 'half_left', 'neutral', 'half_right', 'right']

label_to_index = dict((name, index) for index,name in enumerate(label_names))
print(label_to_index)

def index_to_name(index):
    return label_names[index]


import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from sklearn.preprocessing import LabelBinarizer
from PIL import Image
import numpy as np
import json

class DonkeyDataset(Dataset):
    
    def __init__(self, data_dir):
        
        # Construct one-hot label binarizer
        lb = LabelBinarizer()
        lb.fit([0, 1, 2, 3, 4])
        
        # Initialize data lists
        self.label_list = []
        self.images = []
        
        # Construct data lists
        data_root = pathlib.Path(data_dir)
        all_json_paths = list(data_root.glob('record_*.json'))
        all_json_paths = [str(path) for path in all_json_paths]
        
        for json_path in all_json_paths:
            with open(json_path) as json_file:
                json_data = json.load(json_file)

                # Process image and append to list
                image_filepath = os.path.join(data_dir, json_data['cam/image_array'])
                img = Image.open(image_filepath)
                img = img.convert('RGB')
                img = transforms.functional.resize(img, (64, 64))
                img = transforms.functional.to_tensor(img)
                self.images.append(img)
                
                label_val = value_to_label_index(json_data['user/angle'])
                self.label_list.append(label_val)
            
        self.labels = lb.transform(self.label_list).astype(np.float32)
        
    def __getitem__(self, index):
        
        img = self.images[index]
        label = torch.from_numpy(self.labels[index])
        
        return img, label
    
    def __len__(self):
        
        return len(self.images)


donkey_dataset = DonkeyDataset(data_dir)
print(len(donkey_dataset))

from matplotlib import pyplot as plt

def show_image_tensor(dataset, index):
    plt.imshow(donkey_dataset[index][0].permute(1, 2, 0))
    plt.show()
    
show_image_tensor(donkey_dataset, 2)


# Show sizes of data
img_tensor = donkey_dataset[0][0]
print(img_tensor.size())


train_loader = DataLoader(donkey_dataset, batch_size=30, shuffle=True)


class SimpleNet(nn.Module):
    
    def __init__(self):
        super(SimpleNet, self).__init__()
                
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=3)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        self.fc1 = nn.Linear(2352, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)
        
        
    def forward(self, x):
        
#         print("Input: {}".format(x))
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
#         print(x.size())
#         print("After conv1 and pooling sum: {}".format(torch.sum(x)))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
#         print(x.size())
#         print("After conv2 and pooling sum: {}".format(torch.sum(x)))
        x = x.view(-1, self.num_flat_features(x))
#         print(x.size())
        x = self.fc1(x)
#         print(x.size())
        x = self.fc2(x)
#         print(x.size())
        x = self.fc3(x)
#         print(x.size())
        
        return F.softmax(x, dim=1)
        
        
    def sigmoid(self, x):
        
        sig = torch.nn.Sigmoid()
        
        return sig(x)
    
    def num_flat_features(self, x):
        
        size = x.size()[1:]
        num_flat_features = 1
        for s in size:
            num_flat_features *= s
            
        return num_flat_features
    
model = SimpleNet()
print(model)


optimizer = optim.SGD(model.parameters(), lr=0.05)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # data, target = data.cuda(async=True), target.cuda(async=True) # On GPU
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.binary_cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


import math

data_size = len(donkey_dataset)
validation_split = .1
split_ind = math.floor((1 - validation_split) * data_size)
data_indices = list(range(data_size))

train_ind, validation_ind = data_indices[0:split_ind], data_indices[split_ind:]


from torch.utils.data.sampler import SubsetRandomSampler

train_sampler = SubsetRandomSampler(train_ind)
validation_sampler = SubsetRandomSampler(validation_ind)

batch_size = 20

train_loader = torch.utils.data.DataLoader(donkey_dataset, batch_size=batch_size, 
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(donkey_dataset, batch_size=batch_size,
                                                sampler=validation_sampler)


def train(epoch, optimizer):
    
    loss_list = []
    
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.binary_cross_entropy(output, target)
    
        loss_list.append(loss.item())
        
        loss.backward()
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
    model.eval()
    total_correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(validation_loader):
        data, target = Variable(data), Variable(target)
        _, target_inds = torch.max(target, dim=1)

        output = model(data)
        _, output_inds = torch.max(output, dim=1)

        num_correct = torch.eq(target_inds, output_inds).sum().item()
        total_correct += num_correct
        total += len(target_inds)
        
    accuracy = total_correct / total * 100
    print("Model Accuracy: {:.2f}%".format(accuracy))
    
    return loss_list, accuracy


lr_list = [0.1], 0.15, 0.2] # , 0.05, 0.1, 0.15, 0.2, 0.25]
loss_matrix = []
accuracy_matrix = []

for learning_rate in lr_list:
    
    print("Training with learning rate: {}".format(learning_rate))
    
    # Reinitialize model
    model = SimpleNet()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Train
    loss_list = []
    accuracy_list = []
    for epoch in range(0, 10):
        loss, accuracy = train(epoch, optimizer)
        loss_list.extend(loss)
        accuracy_list.append(accuracy)
        
    loss_matrix.append(loss_list)
    accuracy_matrix.append(accuracy_list)


plt.rcParams['figure.figsize'] = [15, 10]

fig, axs = plt.subplots(2)

legend = []

for i in range(0, len(lr_list)):
    axs[0].plot(list(range(len(loss_matrix[i]))), loss_matrix[i])
    axs[0].set_title("Loss Convergence")
    axs[0].set(xlabel='Batch', ylabel='Loss')
    axs[1].plot(list(range(len(accuracy_matrix[i]))), accuracy_matrix[i])
    axs[1].set_title("Accuracy")
    axs[1].set(xlabel='Epoch', ylabel='Loss')
    legend.append('{}'.format(lr_list[i]))
    
plt.legend(legend, loc='upper right')
plt.show()


test_loader = DataLoader(donkey_dataset, batch_size=1, shuffle=True)
dataiter = iter(test_loader)
images, labels = dataiter.next()

# print(images.size())

output = model(images)

_, result_value = torch.max(output, dim=1)
_, label_value = torch.max(labels, dim=1)

result_name = index_to_name(result_value)
label_name = index_to_name(label_value)

# print("Prediction: {}".format(result_name))
# print("Correct Label: {}".format(label_name))

def show_image(images, index, result_value, label_value):
    plt.imshow(images[index].permute(1, 2, 0))
    result_name = index_to_name(result_value)
    label_name = index_to_name(label_value)
    plt.title("Prediction: {}\nCorrect Label: {}".format(result_name, label_name))
    plt.show()
    
show_image(images, 0, result_value, label_value)


