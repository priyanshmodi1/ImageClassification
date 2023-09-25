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


#image path (defualt same as notebook)
parser.add_argument('dest', default='flowers/test/93/image_06037.jpg', action="store", type = str)

#directory
parser.add_argument('--dir', action="store",dest="data_dir", default="./flowers/")

#gpu
parser.add_argument('--gpu', action="store", default="cuda", help = "set to cuda or cpu")


#top classes to display(5)
parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)

#using cat_to_name.json file
parser.add_argument('--cat_names', dest="category_names", action="store", default='cat_to_name.json')

#checkpoint
parser.add_argument('checkpoint', default='checkpoint.pth', nargs='?', action="store", type = str)



ppa = parser.parse_args()

image = ppa.dest

topk = ppa.top_k

gpu = ppa.gpu

jn = ppa.cat_names

save_dir = ppa.checkpoint

def main():
    model=load_cp(save_dir)
    with open(jn, 'r') as f:
        spec = json.load(f)
        
        
    arprob = predict(image, model, topk, gpu)
    probb = np.array(arprob[0][0])
    labels = [spec[str(i + 1)] for i in np.array(arprob[1][0])]
    
    i = 0
    for i in range(topk)
        print("label: {} --->  probability{}".format(labels[i], prob[i]))
        i += 1
    

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
        comp = torch.device("cpu"
    
    
    
    
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
  
    
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    model = model.to(comp)
                            
    #check gpu once again 
    if torch.cuda.is_available() and comp == 'gpu':
        comp = torch.device("cuda")
    else:
        comp = torch.device("cpu")
    model.to(comp)


    
def load_cp(save_dir = 'checkpoint.pth'):
    #for p in model.parameters():
       #p.requires_grad = False
     
     
    cp = torch.load(save_dir)
    arch = cp['arch']
    lr = cp['lr']
    hu = cp['hu']
    dp = cp['dp']
    epochs = cp['epochs']
    

    model = network(arch, dp, hu, lr)
    
    model.class_to_idx = cp['class_to_idx']
    model.load_state_dict(cp['state_dict'])
    
    return model

def predict(image, model, topk=5, device='cpu'):   
      
    model.eval()
    model.to(device)
    pic = process_image(image_path)
    pic = pic.unsqueeze(0)
    with torch.no_grad():
        final = model.forward(pic)
        high_prob, g = torch.topk(final, topk)
        high_prob = high_prob.exp()   
    pic_dict = {model.class_to_idx[key]: key for key in model.class_to_idx}
    fill_map = list()
    
    g = g.numpy()[0]
    
    for label in g:
        fill_map.append(pic_dict[label])
    return g, fill_map
    
def process_image(image):
    
    
    pil_load = Image.open(image).convert("RGB")
    transform = trans.Compose([trans.Resize(256), trans.CenterCrop(224),trans.ToTensor(),
                             trans.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    image = transform(pil_load)
    return image

    
if __name__== "__main__":
    main()