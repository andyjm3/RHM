import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from torch import nn
import torch
import math
import pickle

import geoopt
from manifolds import *
from geoopt.tensor import ManifoldParameter, ManifoldTensor



#%%
# References: 
# https://github.com/deepmind/symplectic-gradient-adjustment/blob/master/Symplectic_Gradient_Adjustment.ipynb
# https://github.com/alexlyzhov/n-player-games/blob/master/gan_experiments.ipynb
# orthogonal WGAN
def kde(mu, tau, bbox=None, xlabel="", ylabel="", cmap='Blues', savedir=None):
    values = np.vstack([mu, tau])
    kernel = stats.gaussian_kde(values)

    fig, ax = plt.subplots()
    ax.axis(bbox)
    ax.set_aspect(abs(bbox[1]-bbox[0])/abs(bbox[3]-bbox[2]))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    xx, yy = np.mgrid[bbox[0]:bbox[1]:300j, bbox[2]:bbox[3]:300j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    f = np.reshape(kernel(positions).T, xx.shape)
    cfset = ax.contourf(xx, yy, f, cmap=cmap)
    #ax.axes.xaxis.set_visible(False)
    #ax.axes.yaxis.set_visible(False)
    ax.axis('off')
    if savedir is not None:
        plt.savefig(savedir, bbox_inches='tight')
    plt.show()


def x_real_builder(batch_size):
    
    sigma = 0.4
    grid = np.array([[0.5, 0.5], [0.5, 0.0], [0.5, -0.5],
                     [0.0, 0.5], [0.0, 0.0], [0.0, -0.5], 
                     [-0.5, 0.5], [-0.5, 0.0], [-0.5, -0.5]])
    
    temp = np.tile(grid, (batch_size // 9 + 1,1))
    
    mus = temp[0:batch_size,:]
    arr = mus + sigma*np.random.randn(batch_size, 2)*.2
    return arr.astype(np.float32)


def compute_hamiltonian_loss(generator, discriminator, loss):
    
    N = 1000
    X_real = (torch.from_numpy(x_real_builder(N)).float()).to(device)
    
    Z = torch.randn((N, z_dim)).to(device)
    
    X_gen = generator(Z)
    
    disc_out_real = discriminator(X_real)
    disc_out_fake = discriminator(X_gen)
    
    # discriminator loss
    disc_loss_real = loss(disc_out_real, torch.ones_like(disc_out_real).to(device))
    disc_loss_fake = loss(disc_out_fake, torch.zeros_like(disc_out_real).to(device))
    disc_loss = disc_loss_real + disc_loss_fake
    # generator loss 
    gen_loss = loss(disc_out_fake, torch.ones_like(disc_out_real).to(device))
    #gen_loss = -disc_loss
    
    # discrimator generator gradient
    disc_grad = torch.autograd.grad(disc_loss, discriminator.parameters(), retain_graph=True)
    gen_grad = torch.autograd.grad(gen_loss, generator.parameters())
    
    hamit = compute_hamiltonian(generator, gen_grad) + compute_hamiltonian(discriminator, disc_grad)
    
    return disc_loss.detach().item(), gen_loss.detach().item(), hamit.detach().item(), X_gen
    

@torch.no_grad()
def compute_hamiltonian(model, grads):
    hamit = 0
    for idx, param in enumerate(model.parameters()):
        if isinstance(param, (ManifoldParameter, ManifoldTensor)):
            mfd = param.manifold
            grad = grads[idx]
            hamit += mfd.norm(param, grad)**2/2
        else:
            grad = grads[idx]
            hamit += (grad*grad).sum()/2
    
    return hamit


@torch.no_grad()
def set_grads(model, grad):
    for idx, param in enumerate(model.parameters()):
        param.grad = grad[idx]
            

@torch.no_grad()
def egrad2rgrad(egrads, parameters):
    """Converting egrad2rgrad for all parameters"""
    
    rgrads = [None] * len(egrads)
    for idx, param in enumerate(parameters):
        if isinstance(param, (ManifoldParameter, ManifoldTensor)):
            mfd = param.manifold
            egrad = egrads[idx]
            rgrads[idx] = mfd.egrad2rgrad(param, egrad).detach()
        else: 
            rgrads[idx] = egrads[idx].detach()
            
    return rgrads


def eval_hvp(params, egrads, rgrads):
    """ Compute gradient and hvp along rgrad """
    #https://github.com/rlworkgroup/garage/blob/master/src/garage/torch/optimizers/conjugate_gradient_optimizer.py
    
    grad_vector_product = torch.sum(torch.stack(
                [torch.sum(g * x) for g, x in zip(egrads, rgrads)]))
    
    hvp_ls = list(torch.autograd.grad(grad_vector_product, params,
                                retain_graph=False))
    
    for i, (hx, p) in enumerate(zip(hvp_ls, params)):
        if hx is None:
            hvp_ls[i] = torch.zeros_like(p)
    
    return hvp_ls





#%%
# define the discriminator and generator
class MLPdiscriminator(nn.Module):
    
    def __init__(self, indim, hdim, outdim, seed=None):
        super().__init__()
        
        self.indim = indim
        self.outdim = outdim
        self.hdim = hdim
        
        # weight on stiefel manifold
        self.Linear1 = nn.Linear(indim, hdim)
        self.Linear2 = nn.Linear(hdim, hdim)
        self.Linear3 = nn.Linear(hdim, hdim)
        self.Linear4 = nn.Linear(hdim, hdim)                
        self.W5 = geoopt.ManifoldParameter(torch.empty(hdim, hdim), 
                                          manifold=Mstiefel)
        self.b5 = torch.nn.Parameter(torch.empty(hdim,))
        self.Linear6 = nn.Linear(hdim, outdim)
        
        self.reset_parameters(seed)

    def forward(self, X):
        Z = torch.relu(self.Linear1(X))
        Z = torch.relu(self.Linear2(Z))
        Z = torch.relu(self.Linear3(Z))
        Z = torch.relu(self.Linear4(Z))
        Z = torch.relu(Z @ self.W5 + self.b5)
        Z = self.Linear6(Z)
        return Z
    
    @torch.no_grad()
    def reset_parameters(self, seed = None):
        if seed is not None:
            torch.manual_seed(seed)
            
        hdim = self.hdim
        
        stdv = 1. / math.sqrt(hdim)
            
        self.W5.data = Mstiefel.random(hdim, hdim).detach().clone().to(device)
        self.b5.data.uniform_(-stdv, stdv).to(device)
        self.Linear1.reset_parameters()
        self.Linear2.reset_parameters()
        self.Linear3.reset_parameters()
        self.Linear4.reset_parameters()
        self.Linear6.reset_parameters()

class MLPgenerator(nn.Module):
    
    def __init__(self, indim, hdim, outdim, seed=None):
        super().__init__()
        
        self.indim = indim
        self.outdim = outdim
        self.hdim = hdim
        
        # weight on stiefel manifold
        self.Linear1 = nn.Linear(indim, hdim)
        self.Linear2 = nn.Linear(hdim, hdim)
        self.Linear3 = nn.Linear(hdim, hdim)
        self.Linear4 = nn.Linear(hdim, hdim)   
        self.Linear5 = nn.Linear(hdim, hdim)  
        self.Linear6 = nn.Linear(hdim, outdim)
        
        self.reset_parameters(seed)

    def forward(self, X):
        Z = torch.relu(self.Linear1(X))
        Z = torch.relu(self.Linear2(Z))
        Z = torch.relu(self.Linear3(Z))
        Z = torch.relu(self.Linear4(Z))
        Z = torch.relu(self.Linear5(Z))
        Z = self.Linear6(Z)
        return Z
    
    @torch.no_grad()
    def reset_parameters(self, seed = None):
        if seed is not None:
            torch.manual_seed(seed)
            
        self.Linear1.reset_parameters()
        self.Linear2.reset_parameters()
        self.Linear3.reset_parameters()
        self.Linear4.reset_parameters()
        self.Linear5.reset_parameters()
        self.Linear6.reset_parameters()







#%%
# define the trainer functions for RSGDA and RHMSGD
def RSGDArun(generator, discriminator, 
             optimizer_g, optimizer_d, 
             batchsize, maxiter=10, 
             checkperiod = 1,
             scheduler_g=None,scheduler_d=None, seed=0):
    
    logs = {}
    logs['iter'] = []
    logs['hamit'] = []
    logs['loss_g'] = []
    logs['loss_d'] = []
    
    loss = nn.BCEWithLogitsLoss()
    
    
    for it in range(maxiter+1):
        
        # checkperiod print and plots
        if it % checkperiod == 0:
            disc_loss_full, gen_loss_full, hamit, X_gen = compute_hamiltonian_loss(generator, discriminator, loss)
            logs['iter'].append(it)
            logs['loss_d'].append(disc_loss_full)
            logs['loss_d'].append(gen_loss_full)
            logs['hamit'].append(hamit)
            
            bbox = [-1, 1, -1, 1]
            #savedir = main_dir + 'RSGDA_{}_{}'.format(it, seed) + '.pdf'
            savedir = None
            
            x_fake_numpy = X_gen.cpu().detach().numpy()
            
            kde(x_fake_numpy[:,0], x_fake_numpy[:,1], bbox, savedir = savedir)
            
            print(f"[RSGDA] Epoch {it}: {hamit:.4f} (hamit) {gen_loss_full:.4f} (loss_g) {disc_loss_full:.4f} (loss_d) {optimizer_g.param_groups[0]['lr']:.4f} (lr_g)")
        
        optimizer_g.zero_grad()
        optimizer_d.zero_grad()
        
        z = torch.randn((batchsize, z_dim)).to(device)
        x_fake = generator(z)
        
        x_real = (torch.from_numpy(x_real_builder(batchsize)).float()).to(device)
        
        disc_out_real = discriminator(x_real)
        disc_out_fake = discriminator(x_fake)
        
        # discriminator generator loss (Normal GAN)
        disc_loss_real = loss(disc_out_real, torch.ones_like(disc_out_real).to(device))
        disc_loss_fake = loss(disc_out_fake, torch.zeros_like(disc_out_fake).to(device))
        disc_loss = disc_loss_real + disc_loss_fake
        gen_loss = -disc_loss 
        
        # discrimator generator gradient
        disc_grad = torch.autograd.grad(disc_loss, discriminator.parameters(), retain_graph=True)
        gen_grad = torch.autograd.grad(gen_loss, generator.parameters())
        
        set_grads(discriminator, disc_grad)
        set_grads(generator, gen_grad)
        
        optimizer_g.step()
        optimizer_d.step()
            
        if scheduler_g is not None:
            scheduler_g.step()
        if scheduler_d is not None:
            scheduler_d.step()
            
        
    return logs



def RHMSGDrun(generator, discriminator, 
             optimizer_g, optimizer_d, 
             batchsize,
             gamma = 0, maxiter=10, 
             checkperiod = 1,
             scheduler_g=None,scheduler_d=None, seed=0):
    
    logs = {}
    logs['iter'] = []
    logs['hamit'] = []
    logs['loss_g'] = []
    logs['loss_d'] = []
    
    loss = nn.BCEWithLogitsLoss()
    bshalf = int(batchsize/2)
    
    for it in range(maxiter+1):
        
        # checkperiod print and plots
        if it % checkperiod == 0:
            disc_loss, gen_loss, hamit, X_gen = compute_hamiltonian_loss(generator, discriminator, loss)
            logs['iter'].append(it)
            logs['loss_d'].append(disc_loss)
            logs['loss_d'].append(gen_loss)
            logs['hamit'].append(hamit)
            
            bbox = [-1, 1, -1, 1]
            #savedir = main_dir + 'RHMSGD_{}_{}'.format(it, seed) + '.pdf'
            savedir = None
            
            x_fake_numpy = X_gen.cpu().detach().numpy()
            
            kde(x_fake_numpy[:,0], x_fake_numpy[:,1], bbox, savedir = savedir)
            
            print(f"[RHM-SGD] Epoch {it}: {hamit:.4f} (hamit) {gen_loss:.4f} (loss_g) {disc_loss:.4f} (loss_d) {optimizer_g.param_groups[0]['lr']:.4f} (lr_g)")
        
        optimizer_g.zero_grad()
        optimizer_d.zero_grad()
        
        
        z = torch.randn((batchsize, z_dim)).to(device)
        z1 = z[:bshalf, :]
        z2 = z[bshalf:, :]
        
        x_fake1 = generator(z1)
        x_fake2 = generator(z2)
        
        x_real = (torch.from_numpy(x_real_builder(batchsize)).float()).to(device)
        x_real1 = x_real[:bshalf, :]
        x_real2 = x_real[bshalf:, :]
        
        disc_out_real1 = discriminator(x_real1)
        disc_out_real2 = discriminator(x_real2)
        disc_out_fake1 = discriminator(x_fake1)
        disc_out_fake2 = discriminator(x_fake2)
        
        # discriminator generator loss
        disc_loss_real1 = loss(disc_out_real1, torch.ones_like(disc_out_real1).to(device))
        disc_loss_real2 = loss(disc_out_real2, torch.ones_like(disc_out_real2).to(device))
        disc_loss_fake1 = loss(disc_out_fake1, torch.zeros_like(disc_out_fake1).to(device))
        disc_loss_fake2 = loss(disc_out_fake2, torch.zeros_like(disc_out_fake2).to(device))
        
        # note the gen_loss = -disc_loss
        disc_loss1 = disc_loss_real1 + disc_loss_fake1
        disc_loss2 = disc_loss_real2 + disc_loss_fake2
        
        
        gen_params = list(generator.parameters())
        dis_params = list(discriminator.parameters())
        
        params = gen_params + dis_params
        egrad1 = torch.autograd.grad(disc_loss1, params, create_graph=True)
        rgrad1 = egrad2rgrad(egrad1, params)
        
        egrad2 = torch.autograd.grad(disc_loss2, params, create_graph=True)
        rgrad2 = egrad2rgrad(egrad2, params)
        
        # hvp
        hvp_ls1 = eval_hvp(params, egrad1, rgrad2)
        hvp_ls2 = eval_hvp(params, egrad2, rgrad1)
    
        # loop through the params and set gradients
        for idx, param in enumerate(generator.parameters()):
            # no manifold param for generator
            hvp1_i = hvp_ls1[:len(gen_params)][idx]#.clone()
            hvp2_i = hvp_ls2[:len(gen_params)][idx]#.clone()
            egrad1_i = egrad1[:len(gen_params)][idx]#.clone()
            egrad2_i = egrad2[:len(gen_params)][idx]#.clone()
            param.grad = (hvp1_i + hvp2_i)/2 - gamma * (egrad1_i + egrad2_i)/2
            
            
        for idx, param in enumerate(discriminator.parameters()):
            if isinstance(param, (ManifoldParameter, ManifoldTensor)):
                mfd = param.manifold
                hvp1_i = hvp_ls1[len(gen_params):][idx]#.clone()
                hvp2_i = hvp_ls2[len(gen_params):][idx]#.clone()
                egrad1_i = egrad1[len(gen_params):][idx]#.clone()
                egrad2_i = egrad2[len(gen_params):][idx]#.clone()
                rgrad1_i = rgrad1[len(gen_params):][idx]#.clone()
                rgrad2_i = rgrad2[len(gen_params):][idx]#.clone()
                hvp1 = mfd.ehess2rhesspreproj(param.data, egrad1_i, hvp1_i, rgrad2_i) # convert to rhess before proj
                hvp2 = mfd.ehess2rhesspreproj(param.data, egrad2_i, hvp2_i, rgrad1_i)
                param.grad = (hvp1 + hvp2)/2 + gamma * (egrad1_i + egrad2_i)/2
            
            else:
                hvp1_i = hvp_ls1[len(gen_params):][idx]#.clone()
                hvp2_i = hvp_ls2[len(gen_params):][idx]#.clone()
                egrad1_i = egrad1[len(gen_params):][idx]#.clone()
                egrad2_i = egrad2[len(gen_params):][idx]#.clone()
                param.grad = (hvp1_i + hvp2_i)/2 + gamma * (egrad1_i + egrad2_i)/2
        
        optimizer_g.step()
        optimizer_d.step()
            
        if scheduler_g is not None:
            scheduler_g.step()
        if scheduler_d is not None:
            scheduler_d.step()
    
    return logs




#%%
if __name__ == '__main__':

    Mstiefel = Stiefel(canonical = False)
    MEuclidean = Sphere()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
            
    
    maxiter = 30000
    batchsize = 256
    checkperiod = 2500
    
    z_dim = 64

    seeds = [9, 42, 101, 1001, 10001]
    
    for seed in seeds:

        print('seed', seed)

        torch.manual_seed(seed)
        np.random.seed(seed)

        # # RSGDA
        rg = 0.008
        rd = 0.008
        generator = MLPgenerator(z_dim, 128, 2, seed=seed).to(device)
        discriminator = MLPdiscriminator(2, 128, 1, seed=seed).to(device)
        optimizer_g = geoopt.optim.RiemannianSGD(generator.parameters(), lr=rg)
        optimizer_d = geoopt.optim.RiemannianSGD(discriminator.parameters(), lr=rd)
        logs_RSGDA = RSGDArun(generator, discriminator, 
                optimizer_g, optimizer_d, 
                batchsize, maxiter, checkperiod, seed=seed)
        
        

        # RHMSGD
        gamma = 0.5
        rg = 0.1
        rd = 0.1
        generator = MLPgenerator(z_dim, 128, 2, seed=seed).to(device)
        discriminator = MLPdiscriminator(2, 128, 1, seed=seed).to(device)
        optimizer_g = geoopt.optim.RiemannianSGD(generator.parameters(), lr=rg)
        optimizer_d = geoopt.optim.RiemannianSGD(discriminator.parameters(), lr=rd)
        logs_RHMSGD = RHMSGDrun(generator, discriminator, 
                optimizer_g, optimizer_d, 
                batchsize,
                gamma, maxiter, 
                checkperiod, seed=seed)
        

