from torch.autograd import Variable
import torch.nn.functional as F
import torch

def train(epoch, model, optimizer, train_loader, validation_loader):
    """
    Trains a model for one epoch
    :param epoch: Epoch number
    :param model: Neural net model 
    :param optimizer: Optimizer with parameters
    :param train_loader: Data loader for training data
    :param validation_loader: Data loader for validation data
    """
    
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
    
    return model, loss_list, accuracy