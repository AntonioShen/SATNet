import torch
import torch.nn as nn

import numpy as np

from tqdm import tqdm
from torchvision.utils import save_image

from pathlib import Path

# Model Hyperparameters

dataset_path = '~/datasets'

cuda = True
DEVICE = torch.device("cuda" if cuda else "cpu")
PATH = 'vae_save.pt'

batch_size = 100

x_dim  = 784
hidden_dim = 400
latent_dim = 9

lr = 1e-3

epochs = 10

from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


mnist_transform = transforms.Compose([
        transforms.ToTensor(),
])

kwargs = {'num_workers': 1, 'pin_memory': True} 

train_dataset = MNIST(dataset_path, transform=mnist_transform, train=True, download=True)
test_dataset  = MNIST(dataset_path, transform=mnist_transform, train=False, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=True,  **kwargs)

class Encoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        
        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_mean  = nn.Linear(hidden_dim, latent_dim)
        self.FC_var   = nn.Linear (hidden_dim, latent_dim)
        self.training = True
        
    def forward(self, x):
        h_       = torch.relu(self.FC_input(x))
        mean     = self.FC_mean(h_)
        log_var  = self.FC_var(h_)                     # encoder produces mean and log of variance 
                                                       #             (i.e., parateters of simple tractable normal distribution "q")
        var      = torch.exp(0.5*log_var)              # takes exponential function
        z        = self.reparameterization(mean, var)
        
        return z, mean, log_var
    
    
    def reparameterization(self, mean, var,):
        epsilon = torch.rand_like(var).to(DEVICE)        # sampling epsilon
        
        z = mean + var*epsilon                          # reparameterization trick
        
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)

        self.FC_reconstruct = nn.Linear(latent_dim, 10)
        
    def forward(self, x):
        h     = torch.relu(self.FC_hidden(x))
        x_hat = torch.sigmoid(self.FC_output(h))

        y_hat = torch.log_softmax(self.FC_reconstruct(x), dim=1)
        return x_hat, y_hat

class Model(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
                
    def forward(self, x):
        z, mean, log_var = self.Encoder(x)
        x_hat, y_hat           = self.Decoder(z)
        
        return x_hat, mean, log_var, y_hat


encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim)

model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)


from torch.optim import Adam

BCE_loss = nn.BCELoss()

def loss_function(x, x_hat, mean, log_var, y, y_hat):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    classification_loss = nn.functional.nll_loss(y_hat, y, reduction='sum')
    KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD + classification_loss, reproduction_loss.item(), KLD.item(), classification_loss.item()


optimizer = Adam(model.parameters(), lr=lr)

if Path(PATH).exists():
    model.load_state_dict(torch.load(PATH))
else:
    print("Start training VAE...")
    model.train()

    for epoch in range(epochs):
        overall_loss = 0
        overall_component_losses = 0, 0, 0
        correct = 0
        for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
            x = x.view(batch_size, x_dim)
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            x_hat, mean, log_var, y_hat = model(x)
            loss, rloss, kloss, closs = loss_function(x, x_hat, mean, log_var, y, y_hat)
            
            overall_loss += loss.item()
            overall_component_losses = overall_component_losses[0] + rloss, overall_component_losses[1] + kloss, overall_component_losses[2] + closs

            pred = y_hat.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(y.view_as(pred)).sum().item()
            
            loss.backward()
            optimizer.step()
            
        print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx*batch_size),"accuracy: ", correct / (batch_idx*batch_size), "Component Losses", [i / (batch_idx*batch_size) for i in overall_component_losses])
        
    print("Finish!!")

    torch.save(model.state_dict(), PATH)

import matplotlib.pyplot as plt

model.eval()

with torch.no_grad():
    for batch_idx, (x, y) in enumerate(tqdm(test_loader)):
        x = x.view(batch_size, x_dim)
        x = x.to(DEVICE)
        
        x_hat, _, _, y_hat = model(x)

        pred = y_hat.argmax(dim=1, keepdim=True).cpu()  # get the index of the max log-probability
        correct = pred.eq(y.view_as(pred)).sum().item()
        print(f'prediceted: {pred[0]} for correct value of {y[0]} ({correct / batch_size} total)')
        
        break

def show_image(x, idx, name):
    x = x.view(batch_size, 28, 28)

    fig = plt.figure()
    # plt.imshow(x[idx].cpu().numpy())
    plt.imsave(name, x[idx].cpu().numpy())

show_image(x, 0, 'target.png')
show_image(x_hat, 0, 'prediction.png')
