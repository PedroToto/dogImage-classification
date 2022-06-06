#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from smdebug import modes
import smdebug.pytorch as smd
from smdebug.profiler.utils import str2bool
from smdebug.pytorch import get_hook

import json
import sys
import logging
import argparse
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


#TODO: Import dependencies for Debugging andd Profiling
# def test(model, test_loader, criterion, hook):

def test(model, test_loader, criterion, hook):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    hook.set_mode(smd.modes.EVAL)
    test_loss=0
    correct=0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs=model(inputs)
            loss=criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            test_loss += loss.item() * inputs.size(0)
            correct += torch.sum(preds == labels.data).item()

    test_loss /= len(test_loader.dataset)
    
    logger.info(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )


def train(model, train_loader, criterion, optimizer, hook,epochs):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.train()
    hook.set_mode(smd.modes.TRAIN)
    loss_counter=0
  
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        running_samples=0
    
        for step, (inputs, labels) in enumerate(train_loader, 1):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
      
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            running_samples+=len(inputs)
          
        
            if running_samples>(0.2*len(train_loader.dataset)):
                break

        epoch_loss = running_loss / running_samples
        epoch_acc = running_corrects / running_samples
              
        logger.info(f"Epoch {epoch}: Loss {epoch_loss}, Accuracy {100*epoch_acc}%")
    
    return model

def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet152(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features
    model.fc = nn.Sequential(
                nn.Linear(num_features, 133),
                nn.LogSoftmax(dim=1)
    )
    return model
    

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    logger.info("Create the data loader")
    train_dir = os.path.join(data, 'train')
    test_dir = os.path.join(data, 'test')
    valid_dir = os.path.join(data, 'valid')
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    test_valid_transform = transforms.Compose([
                                     transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    trainSet = ImageFolder(root=train_dir, transform=train_transform)
    testSet = ImageFolder(root=test_dir, transform=test_valid_transform)
    validSet = ImageFolder(root=valid_dir, transform=test_valid_transform)
    
    train_loader = DataLoader(dataset=trainSet, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=testSet, batch_size=batch_size)
    valid_loader = DataLoader(dataset=validSet, batch_size=batch_size)
    
    
    return {'train_loader': train_loader, 'test_loader': test_loader, 'valid_loader': valid_loader}

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    hook = get_hook(create_if_not_exists=True)
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_func = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''

    logger.info(f"Hyperparameter values: Batch size {args.batch_size}: Number of epochs {args.epochs}, Learning rate {args.lr}")
    data =  create_data_loaders(args.data_dir, args.batch_size)
    model=train(model, data['train_loader'], loss_func, optimizer, hook,args.epochs)
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, data['test_loader'], loss_func, hook)
    
    '''
    TODO: Save the trained model
    '''
    logger.info("Saving the model.")
    with open(os.path.join(args.model_dir, 'model.pth'), 'wb') as f:
        torch.save(model.state_dict(), f)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        metavar="N",
        help="Input batch size for the training (default:8)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        metavar="N",
        help="Number of epochs to train (default: 2)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        metavar="LR",
        help="Learning rate (default: 0.1)"
    )
    
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=50,
        metavar="N",
        help="input batch size for testing (default: 50)",
    )

    
    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    
    args=parser.parse_args()
    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    
    
    main(args)
