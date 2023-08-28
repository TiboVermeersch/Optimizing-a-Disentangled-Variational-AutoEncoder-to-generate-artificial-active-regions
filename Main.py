"Main"

import torch
from torchvision import transforms, datasets
import Model1 as Model
import Train
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats 

# In[1]: Enable CPUs for faster training the model
device = "cuda" if torch.cuda.is_available() else "cpu"
kwargs = {'num_workers': 0, 'pin_memory': True} if device=='cuda' else {}
torch.cuda.empty_cache()

torch.cuda.memory_summary(device=None, abbreviated=False)

# In[2]: Load the data

path_dataset = 'C:\\Users\\Tibo\\Documents\\Thesis Tibo\\images'
transformation = transforms.Compose([transforms.Resize((128, 256)), transforms.ToTensor()])
dataset = datasets.ImageFolder(root=path_dataset, transform=transformation)

trainingset, testset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.99), len(dataset)-int(len(dataset)*0.99)])

test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, drop_last=True, pin_memory=False)

# In[3]: Train the model

latent_dimension = 256
beta = 0.5
num_epochs = 100

vae = Model.VAE(latent_dimension=latent_dimension, beta=beta)
vae = Train.cross_validate_vae(vae, trainingset, num_epochs, beta, k_folds=10, latent_dimension=latent_dimension)

with torch.no_grad():
    for x, _ in test_loader:
        x = x.to(device)
        x = torch.autograd.Variable(x)
        x_hat,_,_ = vae(x)
        break

torch.save(vae.state_dict(), 'C:\\Users\\Tibo\\Documents\\Thesis Tibo\\Results\\ResultsAfter\\SavedModel')

_, mu, lv = vae.forward(x)
z = vae.reparameterize(mu,lv)

z2 = torch.clone(z)
beg=-1.8
end=1.8
stp=0.4
znew = z2[0]

for i in np.arange(beg,end,stp):
    for j in np.arange(beg,end,stp):
        zinit = torch.clone(z2[0])
        zinit[:int(latent_dimension/2)] += i
        zinit[int(latent_dimension/2):int(latent_dimension)] += j
        znew = torch.cat((znew, zinit), 0)

znew = znew.reshape((-1,latent_dimension))
tnew = vae.decode(znew)
lx = len(np.arange(beg,end,stp))
ly = len(np.arange(beg,end,stp))
fig, axes = plt.subplots(nrows=ly, ncols=lx, figsize = (16,16), sharex=True, sharey=True)
plt.subplots_adjust(wspace=0, hspace=-0.5)
k=0

for i in range(lx):
    for j in range(ly):
        axes[i][j].imshow(tnew[k].permute((1,2,0)).detach().cpu().numpy(), interpolation='lanczos')
        k+=1
plt.show()

