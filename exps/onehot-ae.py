import torch
import torch.nn as nn

import numpy as np

from tqdm import tqdm
from torchvision.utils import save_image

from pathlib import Path

import matplotlib.pyplot as plt



def show_prediction(test_loader, model):

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(tqdm(test_loader)):
            x = x.view(batch_size, x_dim)
            x = x.to(DEVICE)
            
            x_hat, z = model(x)


            print(f'z value: {z[0]}')
            print(f'visualized example to target.png and prediction.png')

            show_image(x, 0, 'target.png')
            show_image(x_hat, 0, 'prediction.png')

            break

def show_image(x, idx, name):
    x = x.view(batch_size, 28, 28)

    fig = plt.figure()
    # plt.imshow(x[idx].cpu().numpy())
    plt.imsave(name, x[idx].cpu().numpy())


# Model Hyperparameters

dataset_path = Path(__file__).parent / '../datasets'

cuda = True
DEVICE = torch.device("cuda" if cuda else "cpu")
PATH = 'onehot-ae-save.pt'

batch_size = 100

x_dim  = 784
hidden_dim = 400
latent_dim = 10

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
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2  = nn.Linear(hidden_dim, latent_dim)
        self.training = True
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=1)

        
        return x
    

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)

        
    def forward(self, x):
        h     = torch.relu(self.FC_hidden(x))
        x_hat = torch.sigmoid(self.FC_output(h))

        return x_hat

class Model(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
                
    def forward(self, x):
        z = self.Encoder(x)
        x_hat           = self.Decoder(z)
        
        return x_hat, z


encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim)

model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)


from torch.optim import SGD


optimizer = SGD(model.parameters(), lr=1e2)

if Path(PATH).exists():
    model.load_state_dict(torch.load(PATH))
else:
    print("Start training VAE...")
    model.train()

    for epoch in range(epochs):
        overall_loss = 0
        correct = 0
        tloader = tqdm(train_loader)
        for batch_idx, (x, y) in enumerate(tloader):
            x = x.view(batch_size, x_dim)
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            x_hat, z = model(x)
            loss = nn.functional.binary_cross_entropy(x_hat, x)
            
            overall_loss += loss.item()

            loss.backward()
            optimizer.step()

            tloader.set_description(f'epoch {epoch} batch {batch_idx} -- loss: {loss:.4f}')

        print(f'epoch {epoch} complete -- loss: {overall_loss/len(tloader)}')
        show_prediction(test_loader, model)
            
        
    print("Finish!!")

    torch.save(model.state_dict(), PATH)


model.eval()




