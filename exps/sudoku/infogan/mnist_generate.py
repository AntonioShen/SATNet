import argparse

import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-load_path', required=True, help='Checkpoint to load path from')
args = parser.parse_args()

from models.mnist_model import Generator, Discriminator, QHead, DHead
from utils import noise_sample

from mapping import get_mapping

# Load the checkpoint file
state_dict = torch.load(args.load_path)

# Set the device to run on: GPU or CPU.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
# Get the 'params' dictionary from the loaded state_dict.
params = state_dict['params']

# Create the generator network.
netG = Generator().to(device)
discriminator = Discriminator().to(device)
netQ = QHead().to(device)
netD = DHead().to(device)

# Load the trained generator weights.
netG.load_state_dict(state_dict['netG'])
discriminator.load_state_dict(state_dict['discriminator'])
netQ.load_state_dict(state_dict['netQ'])
netD.load_state_dict(state_dict['netD'])

c = np.linspace(-2, 2, 9).reshape(1, -1)
c = np.repeat(c, 10, 0).reshape(-1, 1)
c = torch.from_numpy(c).float().to(device)
c = c.view(-1, 1, 1, 1)

zeros = torch.zeros(90, 1, 1, 1, device=device)

# Continuous latent code.
c2 = torch.cat((c, zeros), dim=1)
c3 = torch.cat((zeros, c), dim=1)

idx = np.arange(9).repeat(10)
dis_c = torch.zeros(90, 9, 1, 1, device=device)
dis_c[torch.arange(0, 90), idx] = 1.0
# Discrete latent code.
c1 = dis_c.view(90, -1, 1, 1)

z = torch.randn(90, 62, 1, 1, device=device)

# To see variation along c2 (Horizontally) and c1 (Vertically)
noise1 = torch.cat((z, c1, c2), dim=1)
# To see variation along c3 (Horizontally) and c1 (Vertically)
noise2 = torch.cat((z, c1, c3), dim=1)


# Generate image.
with torch.no_grad():
    generated_img1 = netG(noise1)
    print(generated_img1.shape)
    output = discriminator(generated_img1)
    print(output.shape)
    q1, q2, q3 = netQ(output)
    print(q1.shape, q2.shape, q3.shape)
    d = netD(output)
    print(d.shape)

    generated_img1 = generated_img1.detach().cpu()

    # noise, idx = noise_sample(params['num_dis_c'], params['dis_c_dim'], params['num_con_c'], params['num_z'], 10, device)
    # print(noise.shape, idx)


    # for j in range(params['num_dis_c']):
    #     exp_q1 = torch.exp(q1[:, j*10 : j*10 + 10])
    #     probs = exp_q1/torch.sum(exp_q1, dim=1, keepdim=True)

    #     _, category = torch.topk(probs, 1)
    #     category = category.squeeze()
    #     print(category)




values, indices, num_datapoints = get_mapping(lambda x: netQ(discriminator(x)), device, params)
print(f'Train correctness: {values.sum() / num_datapoints} -- mapping: {indices}')

values, indices, num_datapoints = get_mapping(lambda x: netQ(discriminator(x)), device, params, train=False)
print(f'Test correctness: {values.sum() / num_datapoints} -- mapping: {indices}')

# Display the generated image.
fig = plt.figure(figsize=(10, 10))
plt.axis("off")
plt.imshow(np.transpose(vutils.make_grid(generated_img1, nrow=10, padding=2, normalize=True), (1,2,0)))
plt.savefig('mnist_generate_1.png')

# Generate image.
with torch.no_grad():
    generated_img2 = netG(noise2).detach().cpu()
# Display the generated image.
fig = plt.figure(figsize=(10, 10))
plt.axis("off")
plt.imshow(np.transpose(vutils.make_grid(generated_img2, nrow=10, padding=2, normalize=True), (1,2,0)))
plt.savefig('mnist_generate_2.png')
