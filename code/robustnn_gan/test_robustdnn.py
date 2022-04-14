import torch
import geoopt
from manifolds import *
import torchvision
from torch import nn
from geoopt.tensor import ManifoldParameter, ManifoldTensor
from torch.optim.lr_scheduler import StepLR

import pickle   


#%%
def compute_hamiltonian(model, criterion, fulldata):
    
    data = fulldata.data
    data = (data.view(*data.shape[:-2],-1).squeeze()).to(device)
    target = (fulldata.targets).to(device)
    
    out = model(data.to(device))        
    loss = criterion(out, target)   
    
    egrads = torch.autograd.grad(loss, model.parameters())
    rgrads = egrad2rgrad(egrads, model)
    
    # compute hamiltonian    
    with torch.no_grad():
        hamit = 0
        for idx, param in enumerate(model.parameters()):
            if isinstance(param, (ManifoldParameter, ManifoldTensor)):
                mfd = param.manifold
                grad = rgrads[idx]
                hamit += mfd.norm(param, grad)**2/2
            else:
                grad = rgrads[idx]
                hamit += (grad*grad).sum()/2
    
    return hamit


def savestats(logs, hamit, it):
    logs['iter'].append(it)
    logs['hamit'].append(hamit.cpu().detach().item())
    
    return logs


@torch.no_grad()
def egrad2rgrad(egrads, model):
    """Converting egrad2rgrad for all parameters"""
    
    rgrads = [None] * len(egrads)
    for idx, param in enumerate(model.parameters()):
        if isinstance(param, (ManifoldParameter, ManifoldTensor)):
            mfd = param.manifold
            egrad = egrads[idx]
            rgrads[idx] = mfd.egrad2rgrad(param, egrad).detach()
        else: 
            rgrads[idx] = egrads[idx].detach()
            
    return rgrads





def eval_hvp(model, egrads, rgrads):
    """ Compute gradient and hvp along rgrad """
    #https://github.com/rlworkgroup/garage/blob/master/src/garage/torch/optimizers/conjugate_gradient_optimizer.py
    
    grad_vector_product = torch.sum(torch.stack(
                [torch.sum(g * x) for g, x in zip(egrads, rgrads)]))
    
    hvp_ls = list(torch.autograd.grad(grad_vector_product, model.parameters(),
                                retain_graph=False))
    
    for i, (hx, p) in enumerate(zip(hvp_ls, model.parameters())):
        if hx is None:
            hvp_ls[i] = torch.zeros_like(p)
    
    return hvp_ls

#%%
# define the main model
class RobustDNN(torch.nn.Module):
    
    def __init__(self, indim, hdim, nclass, seed=None):
        super().__init__()
        
        self.indim = indim
        self.hdim = hdim
        self.nclass = nclass
        
        # weight on stiefel manifold
        self.W1 = geoopt.ManifoldParameter(torch.empty(indim, hdim), 
                                          manifold=Mstiefel)
        self.W2 = geoopt.ManifoldParameter(torch.empty(hdim, hdim), 
                                          manifold=Mstiefel)
        
        # perturbatation on sphere manifold
        self.p = geoopt.ManifoldParameter(torch.empty(indim,), 
                                          manifold=Msphere)
        
        # bias
        self.b1 = torch.nn.Parameter(torch.empty(hdim,))
        self.b2 = torch.nn.Parameter(torch.empty(hdim,))
        
        self.Linear = nn.Linear(hdim, nclass)
        self.reset_parameters(seed)
        
    
    def forward(self, X):
        # first layer
        Z = (X + self.p) @ self.W1  + self.b1
        Z = torch.relu(Z)
        
        # second layer
        Z = Z @ self.W2 + self.b2
        Z = torch.relu(Z)
        
        # final layer
        Z = self.Linear(Z)
        
        return Z
        
    @torch.no_grad()
    def reset_parameters(self, seed = None):
        # ensure the init is the same
        if seed is not None:
            torch.manual_seed(seed)
        
        indim = self.indim
        hdim = self.hdim
        nclass = self.nclass
            
        self.W1.data = Mstiefel.random(indim, hdim).detach().clone().to(device)
        self.W2.data = Mstiefel.random(hdim, hdim).detach().clone().to(device)
        self.p.data = Msphere.random(indim).detach().clone().to(device)
        self.b1.data = (torch.zeros((hdim,))).to(device)
        self.b2.data = (torch.zeros((hdim,))).to(device)
        self.Linear.weight.data = torch.nn.init.xavier_uniform_(torch.empty(nclass,hdim)).detach().clone().to(device)
        self.Linear.bias.data = (torch.zeros((nclass,))).to(device)




#%%
# define the trainer functions for RSGDA and RHMSGD
def RHMSGDrun(model, criterion, optimizer, dataloader, gamma=0, maxiter=10, scheduler=None):
    
    logs = {}
    logs['iter'] = []
    logs['hamit'] = []
    
    for it in range(maxiter):
        
        hamit = compute_hamiltonian(model, criterion, mnist_data)
        logs = savestats(logs, hamit, it)
        print(f"[RHM-SGD] Epoch {it}: {hamit.item():.4f} (hamit) {optimizer.param_groups[0]['lr']:.4f} (lr)")
        
        for (data, target) in dataloader: 
        
            optimizer.zero_grad()
                
            data = data.view(*data.shape[:-2],-1).squeeze().to(device) # convert to [b, dim]
            target = target.to(device)

            nb = data.shape[0]
            hfnb = int(nb/2)
            
            data1 = data[:hfnb, :]
            data2 = data[hfnb:, :]
            target1 = target[:hfnb]
            target2 = target[hfnb:]
            
            out1 = model(data1)        
            loss1 = criterion(out1, target1)   
            egrad1 = torch.autograd.grad(loss1, model.parameters(), create_graph=True)
            rgrad1 = egrad2rgrad(egrad1, model)
            
            out2 = model(data2)        
            loss2 = criterion(out2, target2)   
            egrad2 = torch.autograd.grad(loss2, model.parameters(), create_graph=True)
            rgrad2 = egrad2rgrad(egrad2, model)
            
            # hvp
            hvp_ls1 = eval_hvp(model, egrad1, rgrad2)
            hvp_ls2 = eval_hvp(model, egrad2, rgrad1)
            
            # loop through the params and set gradients
            for idx, (name, param) in enumerate(model.named_parameters()):
                if isinstance(param, (ManifoldParameter, ManifoldTensor)):
                    mfd = param.manifold
                    hvp1_i = hvp_ls1[idx]#.clone()
                    hvp2_i = hvp_ls2[idx]#.clone()
                    egrad1_i = egrad1[idx]#.clone()
                    egrad2_i = egrad2[idx]#.clone()
                    rgrad1_i = rgrad1[idx]#.clone()
                    rgrad2_i = rgrad2[idx]#.clone()
                    hvp1 = mfd.ehess2rhesspreproj(param.data, egrad1_i, hvp1_i, rgrad2_i) # convert to rhess before proj
                    hvp2 = mfd.ehess2rhesspreproj(param.data, egrad2_i, hvp2_i, rgrad1_i)
                    param.grad = (hvp1 + hvp2)/2
                    if name == 'p':
                        param.grad = param.grad - gamma * (egrad1_i + egrad2_i)/2
                    else:
                        param.grad = param.grad + gamma * (egrad1_i + egrad2_i)/2
                    
                else:
                    egrad1_i = egrad1[idx].clone()
                    egrad2_i = egrad2[idx].clone()
                    param.grad = (hvp_ls1[idx].clone() + hvp_ls2[idx].clone())/2 + gamma * (egrad1_i + egrad2_i)/2
            
            optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
    
    return logs



def RSGDArun(model, criterion, optimizer, dataloader, maxiter=10, scheduler=None):
    
    logs = {}
    logs['iter'] = []
    logs['hamit'] = []
    
    for it in range(maxiter):
        
        hamit = compute_hamiltonian(model, criterion, mnist_data)
        logs = savestats(logs, hamit, it)
        print(f"[RSGDA] Epoch {it}: {hamit.item():.4f} (hamit) {optimizer.param_groups[0]['lr']:.4f} (lr)")
        
        for (data, target) in dataloader: 
        
            optimizer.zero_grad()
                
            data = data.view(*data.shape[:-2],-1).squeeze().to(device) # convert to [b, dim]
            target = target.to(device)
            
            out = model(data)        
            loss = criterion(out, target)   
            loss.backward()
            
            # modify the gradients for p
            for idx, (name, param) in enumerate(model.named_parameters()):
                if name == 'p':
                    param.grad = -param.grad
            
            optimizer.step()
            
        if scheduler is not None:
            scheduler.step()
            
    
    return logs



if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    Mstiefel = Stiefel(canonical = False)
    Msphere = Sphere()
    
    # load mnist data
    mnist_data = torchvision.datasets.MNIST('./data/', download =True,
                                            transform=torchvision.transforms.Compose([
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(
                                                    (0.1307,), (0.3081,))
                                                    ]))
    
    
    # parameters
    maxiter = 30
    
    indim = 784
    hdim = 16
    nclass = 10
    
    seeds = [9, 42, 101, 1001, 10001]
        
    
    for seed in seeds:
        print(seed)
        torch.manual_seed(seed)
    
        train_loader = torch.utils.data.DataLoader(mnist_data, batch_size=64, shuffle=True)
    
        # RHMSGD
        lr_rhmsgd = 0.05
        model = RobustDNN(indim, hdim, nclass, seed).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = geoopt.optim.RiemannianSGD(model.parameters(), lr=lr_rhmsgd)
        log_RSHGD_fix = RHMSGDrun(model, criterion, optimizer, train_loader, gamma=0, maxiter=maxiter) 
    
    
        # RSGDA
        lr_sgda = 0.04
        model = RobustDNN(indim, hdim, nclass, seed).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = geoopt.optim.RiemannianSGD(model.parameters(), lr=lr_sgda)
        log_RSGDA_fix = RSGDArun(model, criterion, optimizer, train_loader, maxiter=maxiter)     




        
            
                
        
        
    








