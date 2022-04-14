# used for plotting final results
import matplotlib.pyplot as plt
import numpy as np
import pickle

col = ['dodgerblue', "tab:orange"]
marker = ['*', 'x']


#%% for robust dnn example
# =============================================================================
# seeds = [9, 42, 101, 1001, 10001]
# 
# 
# RSGDAres = np.zeros((len(seeds), 30))
# RHMSGDres = np.zeros((len(seeds), 30))
# 
# for idx, seed in enumerate(seeds):
#     
#     
#     # gda
#     filestr = './Experiments/robustnn/logsRSGDA_{}'.format(seed) + '.pkl'
#     with open(filestr, 'rb') as f:
#         gda_res = pickle.load(f)
#     
#     gda_hamit = np.array(gda_res['hamit'])
#     gda_hamit_norm = gda_hamit / gda_hamit[0]
#     RSGDAres[idx, :] = gda_hamit_norm
#     
#     # rhm
#     filestr = './Experiments/robustnn/logsRHMSGD_{}'.format(seed) + '.pkl'
#     with open(filestr, 'rb') as f:
#         rhm_res = pickle.load(f)
# 
#     rhm_hamit = np.array(rhm_res['hamit'])
#     rhm_hamit_norm = rhm_hamit / rhm_hamit[0]
#     RHMSGDres[idx, :] = rhm_hamit_norm
#     
# 
# 
# scale = 1
# 
# plt.rcParams["figure.figsize"] = (7,6)
# 
# plt.figure(1)
# plt.yscale("log")
# # gda
# mean_gda = np.mean(RSGDAres,0)
# std_gda = np.std(RSGDAres,0)
# mean_log_gda = np.mean(np.log10(RSGDAres),0)
# std_log_gda = np.std(np.log10(RSGDAres), 0)
# plt.plot(np.arange(len(mean_log_gda)), np.power(10,mean_log_gda), label='RSGDA (4e-2)', color=col[0], marker=marker[0])
# plt.fill_between(np.arange(len(mean_log_gda)), np.power(10,mean_log_gda-scale*std_log_gda) , np.power(10,mean_log_gda+scale*std_log_gda), alpha=0.5, fc=col[0])
# 
# # rhm
# mean_rhm = np.mean(RHMSGDres,0)
# std_rhm = np.std(RHMSGDres,0)
# mean_log_rhm = np.mean(np.log10(RHMSGDres),0)
# std_log_rhm = np.std(np.log10(RHMSGDres), 0)
# plt.plot(np.arange(len(mean_log_rhm)), np.power(10,mean_log_rhm), label='RHM-SGD (5e-2)', color=col[1], marker=marker[1])
# plt.fill_between(np.arange(len(mean_log_rhm)), np.power(10,mean_log_rhm-scale*std_log_rhm) , np.power(10,mean_log_rhm+scale*std_log_rhm), alpha=0.5, fc=col[1])
# 
# plt.legend(prop={'size': 18})
# plt.xlabel("Epoch", fontsize=25)
# plt.xticks(fontsize=15)
# plt.ylabel("Ht / H0", fontsize=25)
# plt.yticks(fontsize=15)
# plt.savefig('./Experiments/robustnn/hamit_iter_robustnn.pdf', bbox_inches='tight')
# 
# =============================================================================




#%% for GAN example


seeds = [9, 42, 101, 1001, 10001]


RSGDAres = np.zeros((len(seeds), 13))
RHMSGDres = np.zeros((len(seeds), 13))

for idx, seed in enumerate(seeds):
    
    # gda
    filestr = './Experiments/gan/logsRSGDA_{}'.format(seed) + '.pkl'
    with open(filestr, 'rb') as f:
        gda_res = pickle.load(f)
    
    gda_hamit = np.array(gda_res['hamit'])
    gda_hamit_norm = gda_hamit / gda_hamit[0]
    RSGDAres[idx, :] = gda_hamit_norm
    
    # rhm
    filestr = './Experiments/gan/logsRHMSGD_{}'.format(seed) + '.pkl'
    with open(filestr, 'rb') as f:
        rhm_res = pickle.load(f)

    rhm_hamit = np.array(rhm_res['hamit'])
    rhm_hamit_norm = rhm_hamit / rhm_hamit[0]
    RHMSGDres[idx, :] = rhm_hamit_norm
    
    iters = np.array(rhm_res['iter'])



scale = 1

plt.rcParams["figure.figsize"] = (7,6)

plt.figure(1)
plt.yscale("log")
# gda
mean_gda = np.mean(RSGDAres,0)
std_gda = np.std(RSGDAres,0)
mean_log_gda = np.mean(np.log10(RSGDAres),0)
std_log_gda = np.std(np.log10(RSGDAres), 0)
plt.plot(iters, np.power(10,mean_log_gda), label='RSGDA (8e-3)', color=col[0], marker=marker[0])
plt.fill_between(iters, np.power(10,mean_log_gda-scale*std_log_gda) , np.power(10,mean_log_gda+scale*std_log_gda), alpha=0.5, fc=col[0])

# rhm
mean_rhm = np.mean(RHMSGDres,0)
std_rhm = np.std(RHMSGDres,0)
mean_log_rhm = np.mean(np.log10(RHMSGDres),0)
std_log_rhm = np.std(np.log10(RHMSGDres), 0)
plt.plot(iters, np.power(10,mean_log_rhm), label='RHM-SCON (1e-1)', color=col[1], marker=marker[1])
plt.fill_between(iters, np.power(10,mean_log_rhm-scale*std_log_rhm) , np.power(10,mean_log_rhm+scale*std_log_rhm), alpha=0.5, fc=col[1])

plt.legend(prop={'size': 18})
plt.xlabel("Iteration", fontsize=25)
plt.xticks(fontsize=15)
plt.ylabel("Ht / H0", fontsize=25)
plt.yticks(fontsize=15)

plt.savefig('./Experiments/gan/hamit_iter_gan.pdf', bbox_inches='tight')
