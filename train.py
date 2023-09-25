#Priyansh Modi
#train.py


import argparse


import matplotlib.pyplot as plt
import torch
import torchvision 
from torch import nn
from torch import optim
import torchvision.transforms as trans
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json 
from collections import OrderedDict
from PIL import Image
import random
import numpy as np

#use argparse***
parser = argparse.ArgumentParser()


#directory
parser.add_argument('--data_dir', action="store", default="flowers")

#checkpoint
parser.add_argument('--save_dir', action="store", default="checkpoint.pth")

#learning rate (0.001)
parser.add_argument('--lr', action="store", type=float, default=0.001)

#epochs (2)
parser.add_argument('--epochs', action = "store", type=int, default=2, help = '# of times model should be trained')

#arch --> must be vgg16/vgg19
parser.add_argument('--arch', action="store", default="vgg16", help = "USE vgg16 or vgg19 to train model")

#hidden units (2048)
parser.add_argument('--hidden_units', action="store", dest="hidden_units", type=int, default=2048)

#whether user wants to use cuda or not (check if cuda is available*)
parser.add_argument('--gpu', action="store", default="cuda", help = "set to cuda or cpu")

#dp rate (0.2)
parser.add_argument('--dp', action="store", type=float, default=0.2, help = "dropout rate")

#setting argparse commands into vars
ppa = parser.parse_args()

data_dir = ppa.data_dir

save_dir = ppa.save_dir

epochs = ppa.epochs

arch = ppa.arch

hu = ppa.hidden_units

gpu = ppa.gpu

lr = ppa.lr

dp = ppa.dp



#set comp to cpu if user chooses or torch.cuda isn't available
if gpu == 'gpu' and torch.cuda.is_available():
    comp = torch.device("cuda")
    
else:
    comp = torch.device("cpu")

    
    
def main():
    train_loader, valid_loader, test_loader, train_data = reader(data_dir)
    model, criterion = network(arch, dp, hu, lr, gpu)
    optimizer = optim.Adam(model.classifier.parameters(), lr= 0.001)
    # Train Model
    steps = 0
    running_loss = 0
    print_every = 5
    print("Network is traning:")
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            steps += 1
          
            if torch.cuda.is_available() and gpu =='gpu':
                inputs, labels = inputs.to(comp), labels.to(comp)
                model = model.to(comp)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        inputs, labels = inputs.to(comp), labels.to(comp)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        valid_loss += batch_loss.item()

                        
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Loss: {running_loss/print_every:.3f}.. "
                      f"Validation Loss: {valid_loss/len(test_loader):.3f}.. "
                      f"Accuracy: {accuracy/len(valid_loader):.3f}")
                running_loss = 0
                model.train()
    
    
    
    
    
    
    
    #save vals
    model.class_to_idx =  train_data.class_to_idx
    torch.save({'arch' :arch, 'hidden_units':hu, 'dp':dp, 'lr':lr,'epochs':epochs, 'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx}, save_dir)
    
#tranforms under 
def reader(d = "flowers"):
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    data_dir = d
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = trans.Compose([trans.RandomRotation(20),
                                      trans.RandomResizedCrop(224),
                                      trans.RandomHorizontalFlip(), 
                                      trans.ToTensor(),
                                      trans.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) 

    valid_transforms = trans.Compose([trans.Resize(255),
                                      trans.CenterCrop(224),
                                      trans.ToTensor(),
                                      trans.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_transforms = trans.Compose([trans.Resize(255),
                                     trans.CenterCrop(224),
                                     trans.ToTensor(),
                                     trans.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms) 
    valid_data = datasets.ImageFolder(valid_dir, transform = valid_transforms) 
    test_data = datasets.ImageFolder(test_dir, transform = test_transforms) 


    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True) 
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size = 64, shuffle = True) 
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = 64, shuffle = True) 

    return train_loader, valid_loader, test_loader, train_data



def network(arch='vgg16', dp=0.2, hu=2048, lr=0.001, comp='gpu'):
    #user must pick between vgg16 and vgg19
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'vgg19':
        model = models.densenet121(pretrained=True)
    else:
        raise TypeError("Use vgg16 or vgg19 model")
        
        
       
    if torch.cuda.is_available():
        comp = torch.device("cuda")
    else:
        comp = torch.device("cpu")
    
    
   
    for p in model.parameters():
                            
        p.requires_grad = False
                            
    model.classifier = nn.Sequential(OrderedDict([ # use sequential to save time 
                          ('fc1', nn.Linear(25088, hu)),
                          ('dropout', nn.Dropout(dp)),
                          ('relu1', nn.ReLU()),
                          ('fc2', nn.Linear(hu, 500)),
                          ('relu2', nn.ReLU()),
                          ('fc3', nn.Linear(500, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
  
    
    print(model)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    model = model.to(comp)
                            
    #check gpu once again 
    if torch.cuda.is_available() and comp == 'gpu':
        comp = torch.device("cuda")
    else:
        comp = torch.device("cpu")
    model.to(comp)

                            
                            
                            
    return model, criterion

def save_cp(train_data, model = 0, save_dir = 'checkpoint.pth', arch = 'vgg16', hu = 2048, dp = 0.2, lr = 0.001, epochs = 2):
    model.class_to_idx =  train_data.class_to_idx
    torch.save({'arch' :arch, 'hu':hu, 'dp':dp, 'lr':lr,'epochs':epochs, 'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx}, save_dir)
   
    

if __name__ == "__main__":
    main()