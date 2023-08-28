"Train"

from sklearn.model_selection import KFold
import numpy as np
import torch
import Model
import matplotlib.pyplot as plt


device = "cuda" if torch.cuda.is_available() else "cpu"
kwargs = {'num_workers': 0, 'pin_memory': True} if device=='cuda' else {}

def train_vae(vae, train_loader, num_epochs, beta):
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4)
    Losses = []
    reconstruction = []
    kl = []

    for epoch in range(num_epochs):
        vae.train()
        total_loss = 0
        total_reconstruction_error = 0
        total_kl_divergence = 0

        for batch_idx, (data_training, _) in enumerate(train_loader):
            # print(batch_idx)
            data_training = data_training.to(device)
            optimizer.zero_grad()

            recon_batch, mu, logvar = vae(data_training)
            loss, reconstruction_error, kl_divergence = vae.loss_function(data_training, recon_batch, mu, logvar)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_reconstruction_error += reconstruction_error.item()
            total_kl_divergence += kl_divergence.item()

        loss_total = total_loss
        Losses.append(loss_total / len(train_loader.dataset))
        reconstruction.append(total_reconstruction_error / len(train_loader.dataset))
        kl.append(total_kl_divergence / len(train_loader.dataset))
        print('Epoch [{}/{}], Loss: {:.3f}'.format(epoch+1, num_epochs, total_loss / len(train_loader.dataset)))

    return Losses, reconstruction, kl

def cross_validate_vae(vae, data, num_epochs, beta, k_folds, latent_dimension):
    kf = KFold(n_splits=k_folds, shuffle=True)
 
    fold_metrics = []

    vae = Model.VAE(latent_dimension, beta).to(device)
    Loss_plot = []
    re_plot = []
    kl_plot = []

    for fold, (train_index, val_index) in enumerate(kf.split(data)):
        print(f"Fold {fold + 1}/{k_folds}") 
        
        train_dataset = torch.utils.data.Subset(data, train_index)
        val_dataset = torch.utils.data.Subset(data, val_index)
 
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

        Loss1, re1, kl1 = train_vae(vae, train_loader, num_epochs, beta)
        for element in Loss1:
            Loss_plot.append(element)

        for element in re1:
            re_plot.append(element)

        for element in kl1:
            kl_plot.append(element)
        
        vae.eval()
        total_loss = 0

        with torch.no_grad():
            for batch_idx, (data_validation, _) in enumerate(val_loader):
                data_validation = data_validation.to(device)
                recon_batch, mu, logvar = vae(data_validation)
                recon_loss = torch.nn.functional.binary_cross_entropy(recon_batch, data_validation, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + beta * kl_loss
                total_loss += loss.item()

        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(data_validation[0].cpu().permute(1, 2, 0), cmap='gray')
        axarr[1].imshow(recon_batch[0].cpu().permute(1, 2, 0), cmap='gray')
        # plt.show()

        fold_metrics.append(total_loss / len(val_dataset))
    

    plt.figure(num='figure_11')
    fig, ax1 = plt.subplots()

    ax1.plot(Loss_plot, label='Total loss')
    ax1.plot(re_plot, color = 'red', label='Reconstruction error')
    ax1.set_ylabel('Loss')

    ax2 = ax1.twinx()
    ax2.plot(kl_plot, color = 'tab:orange', label='KL-divergence')
    ax2.set_ylabel('KL-divergence')
    ax2.set_xlabel('Epochs')

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    ax2.set_title('Loss Curve')
    
    plt.show()

    return vae