# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:40:15 2024

@author: s4479813
"""

import numpy as np
from configobj import ConfigObj
import copy
import emcee
from chainconsumer import ChainConsumer

def format_pardict(pardict):

    pardict["do_corr"] = int(pardict["do_corr"])
    pardict["do_marg"] = int(pardict["do_marg"])
    pardict["do_hex"] = int(pardict["do_hex"])
    pardict["taylor_order"] = int(pardict["taylor_order"])
    pardict["xfit_min"] = np.array(pardict["xfit_min"]).astype(float)
    pardict["xfit_max"] = np.array(pardict["xfit_max"]).astype(float)
    pardict["order"] = int(pardict["order"])
    pardict["scale_independent"] = True if pardict["scale_independent"].lower() is "true" else False
    pardict["z_pk"] = np.array(pardict["z_pk"], dtype=float)
    if not any(np.shape(pardict["z_pk"])):
        pardict["z_pk"] = [float(pardict["z_pk"])]

    return pardict

def read_chain_backend(chainfile):

    #Read the MCMC chain
    reader = emcee.backends.HDFBackend(chainfile)

    #Find the autocorrelation time. 
    tau = reader.get_autocorr_time()
    #Using the autocorrelation time to figure out the burn-in. 
    burnin = int(2 * np.max(tau))
    #Retriving the chain and discard the burn-in, 
    samples = reader.get_chain(discard=burnin, flat=True)
    #The the log-posterior of the chain. 
    log_prob_samples = reader.get_log_prob(discard=burnin, flat=True)
    #Find the best-fit parameters. 
    bestid = np.argmax(log_prob_samples)

    return samples, copy.copy(samples[bestid]), log_prob_samples


#######################              General plotting scheme               ###################################################################
c = ChainConsumer()
chainfile = 'Location of your chain'
sample, max_log_likelihood, log_likelihood = read_chain_backend(chainfile)

#For Full-Modelling
parameters = [r"$\ln(10^{10} A_s)$", r"$h$", r"$\Omega_{\mathrm{cdm}} h^2$", r"$\Omega_bh^2$"] #For LCDM 
# parameters = [r"$\ln(10^{10} A_s)$", r"$h$", r"$\Omega_{\mathrm{cdm}} h^2$", r"$\Omega_bh^2$", r"$w$"] #For wCDM
# parameters = [r"$\ln(10^{10} A_s)$", r"$h$", r"$\Omega_{\mathrm{cdm}} h^2$", r"$\Omega_bh^2$", r"$\Omega_k$"] #For oCDM

c.add_chain(sample[:, :4], parameters = parameters, name = 'Some name for your chain', color = 'Some fancy color') #For LCDM
# c.add_chain(sample[:, :5], parameters = parameters, name = 'Some name for your chain', color = 'Some fancy color') #For wCDM and oCDM

#The fiducial cosmology of the mock
truth = [3.0364, 0.6736, 0.12, 0.02237] #For LCDM
# truth = [3.0364, 0.6736, 0.12, 0.02237, -1.0] #For wCDM
# truth = [3.0364, 0.6736, 0.12, 0.02237, 0.0] #For oCDM

#Plot the chain 
c.plotter.plot(filename='Some_fancy_file_name.png', truth=truth)

#For Shapefit
rd_fid = 147.09681722431307*0.6736 #The sound horizon in Mpc/h with the fidcuial cosmology
radius = 8.0 #We use sigma8 here, so the radius is 8 Mpc/h. 
a_m = 0.6 #The parameter a in Shapefit. It is fixed to 0.6 in our fit. 

sample[:, 2] = sample[:, 2]*np.exp(sample[:, 3]/(2.0*a_m)*np.tanh(a_m*np.log(rd_fid/radius))) #Convert fsigma8 to fsigma_s8 for plotting purpose. 

parameters = [r"$\alpha_{\perp}$", r"$\alpha_{\parallel}$", r"$r_A$", r"$m$"]

c.add_chain(sample[:, :4], parameters = parameters, name = 'Some name for your chain', color = 'Some fancy color')

#The truth values for the Shapefit parameters
truth = [1.0, 1.0, fsigma8_fid, 0.0]

#Plot the chain 
c.plotter.plot(filename='Some_fancy_file_name.png', truth=truth)

############################           The following codes are for the plots in the pybird paper                                 ##############

# ##############                       For figure 5                       ##########################################
# kmax = 0.14 
# rd_fid = 147.09681722431307*0.6736
# radius = 8.0
# a_m = 0.6
# c = ChainConsumer()
# for i in range(8):
#     kmax_new = round(kmax + 0.02*i, 2)
#     try:
#         chainfile = '../../data/DESI_KP4_LRG_pk_' + str(kmax_new) + 'hex' + str(kmax_new) + '_3order_nohex_marg_Shapefit_mock_mean_single_BOSS_MaxF_bin_0.hdf5'
#         sample_new, log_likelihood_new, max_log_likelihood_new = read_chain_backend(chainfile)
#     except:
#         chainfile = '../../data/DESI_KP4_LRG_pk_' + str(kmax_new) + '0hex' + str(kmax_new) + '0_3order_nohex_marg_Shapefit_mock_mean_single_BOSS_MaxF_bin_0.hdf5'
#         sample_new, log_likelihood_new, max_log_likelihood_new = read_chain_backend(chainfile)
#     print(chainfile)
#     sample_new[:, 2] = sample_new[:, 2]*np.exp(sample_new[:, 3]/(2.0*a_m)*np.tanh(a_m*np.log(rd_fid/radius)))
#     sample_new[:, 2] = sample_new[:, 2]/0.450144
    
#     sample_new[:, 4] = sample_new[:, 0]**(2.0/3.0)*sample_new[:, 1]**(1.0/3.0)
#     sample_new[:, 5] = sample_new[:, 1]/sample_new[:, 0]
#     c.add_chain(sample_new[:, :6], parameters = [r"$\alpha_{\perp}$", r"$\alpha_{\parallel}$", r"$r_A$", r"$m$", r"$\alpha_{\mathrm{iso}}$", r"$\alpha_{\mathrm{ap}}$"], name = 'BOSS MaxF', color = 'c')
    
# data_LRG = c.analysis.get_summary()

# kmax = 0.14 
# rd_fid = 147.09681722431307*0.6736
# radius = 8.0
# a_m = 0.6
# c = ChainConsumer()
# for i in range(8):
#     kmax_new = round(kmax + 0.02*i, 2)
#     try:
#         chainfile = '../../data/DESI_KP4_ELG_pk_' + str(kmax_new) + 'hex' + str(kmax_new) + '_3order_nohex_marg_Shapefit_mock_mean_single_BOSS_MaxF_bin_1.hdf5'
#         sample_new, log_likelihood_new, max_log_likelihood_new = read_chain_backend(chainfile)
#     except:
#         chainfile = '../../data/DESI_KP4_ELG_pk_' + str(kmax_new) + '0hex' + str(kmax_new) + '0_3order_nohex_marg_Shapefit_mock_mean_single_BOSS_MaxF_bin_1.hdf5'
#         sample_new, log_likelihood_new, max_log_likelihood_new = read_chain_backend(chainfile)
#     print(chainfile)
#     sample_new[:, 2] = sample_new[:, 2]*np.exp(sample_new[:, 3]/(2.0*a_m)*np.tanh(a_m*np.log(rd_fid/radius)))
#     sample_new[:, 2] = sample_new[:, 2]/0.41754217
    
#     sample_new[:, 4] = sample_new[:, 0]**(2.0/3.0)*sample_new[:, 1]**(1.0/3.0)
#     sample_new[:, 5] = sample_new[:, 1]/sample_new[:, 0]
#     c.add_chain(sample_new[:, :6], parameters = [r"$\alpha_{\perp}$", r"$\alpha_{\parallel}$", r"$r_A$", r"$m$", r"$\alpha_{\mathrm{iso}}$", r"$\alpha_{\mathrm{ap}}$"], name = 'BOSS MaxF', color = 'c')
    
# data_ELG = c.analysis.get_summary()

# kmax = 0.14 
# rd_fid = 147.09681722431307*0.6736
# radius = 8.0
# a_m = 0.6
# c = ChainConsumer()
# for i in range(8):
#     kmax_new = round(kmax + 0.02*i, 2)
#     try:
#         chainfile = '../../data/DESI_KP4_QSO_pk_' + str(kmax_new) + 'hex' + str(kmax_new) + '_3order_nohex_marg_Shapefit_mock_mean_single_BOSS_MaxF_bin_2.hdf5'
#         sample_new, log_likelihood_new, max_log_likelihood_new = read_chain_backend(chainfile)
#     except:
#         chainfile = '../../data/DESI_KP4_QSO_pk_' + str(kmax_new) + '0hex' + str(kmax_new) + '0_3order_nohex_marg_Shapefit_mock_mean_single_BOSS_MaxF_bin_2.hdf5'
#         sample_new, log_likelihood_new, max_log_likelihood_new = read_chain_backend(chainfile)
#     print(chainfile)
#     sample_new[:, 2] = sample_new[:, 2]*np.exp(sample_new[:, 3]/(2.0*a_m)*np.tanh(a_m*np.log(rd_fid/radius)))
#     sample_new[:, 2] = sample_new[:, 2]/0.383341
    
#     sample_new[:, 4] = sample_new[:, 0]**(2.0/3.0)*sample_new[:, 1]**(1.0/3.0)
#     sample_new[:, 5] = sample_new[:, 1]/sample_new[:, 0]
#     c.add_chain(sample_new[:, :6], parameters = [r"$\alpha_{\perp}$", r"$\alpha_{\parallel}$", r"$r_A$", r"$m$", r"$\alpha_{\mathrm{iso}}$", r"$\alpha_{\mathrm{ap}}$"], name = 'BOSS MaxF', color = 'c')
    
# data_QSO = c.analysis.get_summary()

# k = np.linspace(0.14, 0.28, 8)

# import matplotlib.pyplot as plt 

# parameters = [r"$\alpha_{\perp}$", r"$\alpha_{\parallel}$", r"$r_A$", r"$m$", r"$\alpha_{\mathrm{iso}}$", r"$\alpha_{\mathrm{ap}}$"]

# fig = plt.figure()
# param = parameters[0]
# fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex = True)
# # fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, sharex = True)
# LRG_mean = []
# LRG_low = []
# LRG_high = []
# ELG_mean = []
# ELG_low = []
# ELG_high = []
# QSO_mean = []
# QSO_low = []
# QSO_high = []
# for j in range(8):
#     LRG_mean.append(data_LRG[j][param][1])
#     LRG_low.append(data_LRG[j][param][1] - data_LRG[j][param][0])
#     LRG_high.append(data_LRG[j][param][2] - data_LRG[j][param][1])
#     ELG_mean.append(data_ELG[j][param][1])
#     ELG_low.append(data_ELG[j][param][1] - data_ELG[j][param][0])
#     ELG_high.append(data_ELG[j][param][2] - data_ELG[j][param][1])
#     QSO_mean.append(data_QSO[j][param][1])
#     QSO_low.append(data_QSO[j][param][1] - data_QSO[j][param][0])
#     QSO_high.append(data_QSO[j][param][2] - data_QSO[j][param][1])


# ax1.errorbar(k-0.0015, LRG_mean, yerr = np.array([LRG_low, LRG_high]), fmt='.', label = 'LRG', capsize = 2.0)
# ax1.errorbar(k, ELG_mean, yerr = np.array([ELG_low, ELG_high]), fmt='.', label = 'ELG', capsize = 2.0)
# ax1.errorbar(k+0.0015, QSO_mean, yerr = np.array([QSO_low, QSO_high]), fmt='.', label = 'QSO', capsize = 2.0)
# ax1.hlines(1.0, xmin = 0.135, xmax = 0.285, linestyle = 'dashed')
# ax1.set_ylabel(param)

# param = parameters[1]
# LRG_mean = []
# LRG_low = []
# LRG_high = []
# ELG_mean = []
# ELG_low = []
# ELG_high = []
# QSO_mean = []
# QSO_low = []
# QSO_high = []
# for j in range(8):
#     LRG_mean.append(data_LRG[j][param][1])
#     LRG_low.append(data_LRG[j][param][1] - data_LRG[j][param][0])
#     LRG_high.append(data_LRG[j][param][2] - data_LRG[j][param][1])
#     ELG_mean.append(data_ELG[j][param][1])
#     ELG_low.append(data_ELG[j][param][1] - data_ELG[j][param][0])
#     ELG_high.append(data_ELG[j][param][2] - data_ELG[j][param][1])
#     QSO_mean.append(data_QSO[j][param][1])
#     QSO_low.append(data_QSO[j][param][1] - data_QSO[j][param][0])
#     QSO_high.append(data_QSO[j][param][2] - data_QSO[j][param][1])


# ax2.errorbar(k-0.0015, LRG_mean, yerr = np.array([LRG_low, LRG_high]), fmt='.', label = 'LRG', capsize = 2.0)
# ax2.errorbar(k, ELG_mean, yerr = np.array([ELG_low, ELG_high]), fmt='.', label = 'ELG', capsize = 2.0)
# ax2.errorbar(k+0.0015, QSO_mean, yerr = np.array([QSO_low, QSO_high]), fmt='.', label = 'QSO', capsize = 2.0)
# ax2.hlines(1.0, xmin = 0.135, xmax = 0.285, linestyle = 'dashed')
# ax2.set_ylabel(param)

# param = parameters[2]
# LRG_mean = []
# LRG_low = []
# LRG_high = []
# ELG_mean = []
# ELG_low = []
# ELG_high = []
# QSO_mean = []
# QSO_low = []
# QSO_high = []
# for j in range(8):
#     LRG_mean.append(data_LRG[j][param][1])
#     LRG_low.append(data_LRG[j][param][1] - data_LRG[j][param][0])
#     LRG_high.append(data_LRG[j][param][2] - data_LRG[j][param][1])
#     ELG_mean.append(data_ELG[j][param][1])
#     ELG_low.append(data_ELG[j][param][1] - data_ELG[j][param][0])
#     ELG_high.append(data_ELG[j][param][2] - data_ELG[j][param][1])
#     QSO_mean.append(data_QSO[j][param][1])
#     QSO_low.append(data_QSO[j][param][1] - data_QSO[j][param][0])
#     QSO_high.append(data_QSO[j][param][2] - data_QSO[j][param][1])


# ax3.errorbar(k-0.0015, LRG_mean, yerr = np.array([LRG_low, LRG_high]), fmt='.', label = 'LRG', capsize = 2.0)
# ax3.errorbar(k, ELG_mean, yerr = np.array([ELG_low, ELG_high]), fmt='.', label = 'ELG', capsize = 2.0)
# ax3.errorbar(k+0.0015, QSO_mean, yerr = np.array([QSO_low, QSO_high]), fmt='.', label = 'QSO', capsize = 2.0)
# ax3.hlines(1.0, xmin = 0.135, xmax = 0.285, linestyle = 'dashed')
# ax3.set_ylabel(param)

# param = parameters[3]
# LRG_mean = []
# LRG_low = []
# LRG_high = []
# ELG_mean = []
# ELG_low = []
# ELG_high = []
# QSO_mean = []
# QSO_low = []
# QSO_high = []
# for j in range(8):
#     LRG_mean.append(data_LRG[j][param][1])
#     LRG_low.append(data_LRG[j][param][1] - data_LRG[j][param][0])
#     LRG_high.append(data_LRG[j][param][2] - data_LRG[j][param][1])
#     ELG_mean.append(data_ELG[j][param][1])
#     ELG_low.append(data_ELG[j][param][1] - data_ELG[j][param][0])
#     ELG_high.append(data_ELG[j][param][2] - data_ELG[j][param][1])
#     QSO_mean.append(data_QSO[j][param][1])
#     QSO_low.append(data_QSO[j][param][1] - data_QSO[j][param][0])
#     QSO_high.append(data_QSO[j][param][2] - data_QSO[j][param][1])


# ax4.errorbar(k-0.0015, LRG_mean, yerr = np.array([LRG_low, LRG_high]), fmt='.', label = 'LRG', capsize = 2.0)
# ax4.errorbar(k, ELG_mean, yerr = np.array([ELG_low, ELG_high]), fmt='.', label = 'ELG', capsize = 2.0)
# ax4.errorbar(k+0.0015, QSO_mean, yerr = np.array([QSO_low, QSO_high]), fmt='.', label = 'QSO', capsize = 2.0)
# ax4.hlines(0.0, xmin = 0.135, xmax = 0.285, linestyle = 'dashed')
# ax4.set_ylabel(param)
# ax4.set_xlabel(r'$k_{\mathrm{max}} (h/\mathrm{Mpc})$')
# ax4.legend(ncol = 3)

# #Uncomment the following block to plot alpha_iso and alpha_ap. 

# # param = parameters[4]
# # LRG_mean = []
# # LRG_low = []
# # LRG_high = []
# # ELG_mean = []
# # ELG_low = []
# # ELG_high = []
# # QSO_mean = []
# # QSO_low = []
# # QSO_high = []
# # for j in range(8):
# #     LRG_mean.append(data_LRG[j][param][1])
# #     LRG_low.append(data_LRG[j][param][1] - data_LRG[j][param][0])
# #     LRG_high.append(data_LRG[j][param][2] - data_LRG[j][param][1])
# #     ELG_mean.append(data_ELG[j][param][1])
# #     ELG_low.append(data_ELG[j][param][1] - data_ELG[j][param][0])
# #     ELG_high.append(data_ELG[j][param][2] - data_ELG[j][param][1])
# #     QSO_mean.append(data_QSO[j][param][1])
# #     QSO_low.append(data_QSO[j][param][1] - data_QSO[j][param][0])
# #     QSO_high.append(data_QSO[j][param][2] - data_QSO[j][param][1])


# # ax5.errorbar(k-0.0015, LRG_mean, yerr = np.array([LRG_low, LRG_high]), fmt='.', label = 'LRG', capsize = 2.0)
# # ax5.errorbar(k, ELG_mean, yerr = np.array([ELG_low, ELG_high]), fmt='.', label = 'ELG', capsize = 2.0)
# # ax5.errorbar(k+0.0015, QSO_mean, yerr = np.array([QSO_low, QSO_high]), fmt='.', label = 'QSO', capsize = 2.0)
# # ax5.hlines(1.0, xmin = 0.135, xmax = 0.285, linestyle = 'dashed')
# # ax5.set_ylabel(param)

# # param = parameters[5]
# # LRG_mean = []
# # LRG_low = []
# # LRG_high = []
# # ELG_mean = []
# # ELG_low = []
# # ELG_high = []
# # QSO_mean = []
# # QSO_low = []
# # QSO_high = []
# # for j in range(8):
# #     LRG_mean.append(data_LRG[j][param][1])
# #     LRG_low.append(data_LRG[j][param][1] - data_LRG[j][param][0])
# #     LRG_high.append(data_LRG[j][param][2] - data_LRG[j][param][1])
# #     ELG_mean.append(data_ELG[j][param][1])
# #     ELG_low.append(data_ELG[j][param][1] - data_ELG[j][param][0])
# #     ELG_high.append(data_ELG[j][param][2] - data_ELG[j][param][1])
# #     QSO_mean.append(data_QSO[j][param][1])
# #     QSO_low.append(data_QSO[j][param][1] - data_QSO[j][param][0])
# #     QSO_high.append(data_QSO[j][param][2] - data_QSO[j][param][1])


# # ax6.errorbar(k-0.0015, LRG_mean, yerr = np.array([LRG_low, LRG_high]), fmt='.', label = 'LRG', capsize = 2.0)
# # ax6.errorbar(k, ELG_mean, yerr = np.array([ELG_low, ELG_high]), fmt='.', label = 'ELG', capsize = 2.0)
# # ax6.errorbar(k+0.0015, QSO_mean, yerr = np.array([QSO_low, QSO_high]), fmt='.', label = 'QSO', capsize = 2.0)
# # ax6.hlines(1.0, xmin = 0.135, xmax = 0.285, linestyle = 'dashed')
# # ax6.set_ylabel(param)
# # ax6.set_xlabel(r'$k_{\mathrm{max}} (h/\mathrm{Mpc})$')


# ################################################                For figure 8                 ###########################################
# kmax = 0.14 
# c = ChainConsumer()
# for i in range(8):
#     kmax_new = round(kmax + 0.02*i, 2)
#     try:
#         chainfile = '../../data/plotting_chains/figure_8_and_table_4/DESI_KP4_LRG_pk_' + str(kmax_new) + 'hex'+ str(kmax_new) + '_3order_nohex_marg_kmin0p02_fewerbias_bin_0_mean_single_BOSS_MaxF.hdf5'
#         sample_new, log_likelihood_new, max_log_likelihood_new = read_chain_backend(chainfile)
#     except:
#         chainfile = '../../data/plotting_chains/figure_8_and_table_4/DESI_KP4_LRG_pk_' + str(kmax_new) + '0hex'+ str(kmax_new) + '0_3order_nohex_marg_kmin0p02_fewerbias_bin_0_mean_single_BOSS_MaxF.hdf5'
#         sample_new, log_likelihood_new, max_log_likelihood_new = read_chain_backend(chainfile)
#     print(chainfile)
#     c.add_chain(sample_new[:, :4], parameters = [r"$\ln(10^{10} A_s)$", r"$h$", r"$\Omega_{\mathrm{cdm}} h^2$", r"$\Omega_bh^2$"], name = str(kmax_new))
# data_LRG = c.analysis.get_summary()

# kmax = 0.14 
# c = ChainConsumer()
# for i in range(8):
#     kmax_new = round(kmax + 0.02*i, 2)
#     try:
#         chainfile = '../../data/plotting_chains/figure_8_and_table_4/DESI_KP4_ELG_pk_' + str(kmax_new) + 'hex'+ str(kmax_new) + '_3order_nohex_marg_kmin0p02_fewerbias_bin_1_mean_single_BOSS_MaxF.hdf5'
#         sample_new, log_likelihood_new, max_log_likelihood_new = read_chain_backend(chainfile)
#     except:
#         chainfile = '../../data/plotting_chains/figure_8_and_table_4/DESI_KP4_ELG_pk_' + str(kmax_new) + '0hex'+ str(kmax_new) + '0_3order_nohex_marg_kmin0p02_fewerbias_bin_1_mean_single_BOSS_MaxF.hdf5'
#         sample_new, log_likelihood_new, max_log_likelihood_new = read_chain_backend(chainfile)
#     print(chainfile)
#     c.add_chain(sample_new[:, :4], parameters = [r"$\ln(10^{10} A_s)$", r"$h$", r"$\Omega_{\mathrm{cdm}} h^2$", r"$\Omega_bh^2$"], name = str(kmax_new))
    
# kmax = 0.14 
# c = ChainConsumer()
# for i in range(8):
#     kmax_new = round(kmax + 0.02*i, 2)
#     try:
#         chainfile = '../../data/plotting_chains/figure_8_and_table_4/DESI_KP4_QSO_pk_' + str(kmax_new) + 'hex'+ str(kmax_new) + '_3order_nohex_marg_kmin0p02_fewerbias_bin_2_mean_single_BOSS_MaxF.hdf5'
#         sample_new, log_likelihood_new, max_log_likelihood_new = read_chain_backend(chainfile)
#     except:
#         chainfile = '../../data/plotting_chains/figure_8_and_table_4/DESI_KP4_QSO_pk_' + str(kmax_new) + '0hex'+ str(kmax_new) + '0_3order_nohex_marg_kmin0p02_fewerbias_bin_2_mean_single_BOSS_MaxF.hdf5'
#         sample_new, log_likelihood_new, max_log_likelihood_new = read_chain_backend(chainfile)
#     print(chainfile)
#     c.add_chain(sample_new[:, :4], parameters = [r"$\ln(10^{10} A_s)$", r"$h$", r"$\Omega_{\mathrm{cdm}} h^2$", r"$\Omega_bh^2$"], name = str(kmax_new))
# data_QSO = c.analysis.get_summary()

# k = np.linspace(0.14, 0.28, 8)

# import matplotlib.pyplot as plt 

# parameters = [r"$\ln(10^{10} A_s)$", r"$h$", r"$\Omega_{\mathrm{cdm}} h^2$", r"$\Omega_bh^2$"]

# fig = plt.figure()
# param = parameters[0]
# fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex = True)
# # fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, sharex = True)
# LRG_mean = []
# LRG_low = []
# LRG_high = []
# ELG_mean = []
# ELG_low = []
# ELG_high = []
# QSO_mean = []
# QSO_low = []
# QSO_high = []
# for j in range(8):
#     LRG_mean.append(data_LRG[j][param][1])
#     LRG_low.append(data_LRG[j][param][1] - data_LRG[j][param][0])
#     LRG_high.append(data_LRG[j][param][2] - data_LRG[j][param][1])
#     ELG_mean.append(data_ELG[j][param][1])
#     ELG_low.append(data_ELG[j][param][1] - data_ELG[j][param][0])
#     ELG_high.append(data_ELG[j][param][2] - data_ELG[j][param][1])
#     QSO_mean.append(data_QSO[j][param][1])
#     QSO_low.append(data_QSO[j][param][1] - data_QSO[j][param][0])
#     QSO_high.append(data_QSO[j][param][2] - data_QSO[j][param][1])


# ax1.errorbar(k-0.0015, LRG_mean, yerr = np.array([LRG_low, LRG_high]), fmt='.', label = 'LRG', capsize = 2.0)
# ax1.errorbar(k, ELG_mean, yerr = np.array([ELG_low, ELG_high]), fmt='.', label = 'ELG', capsize = 2.0)
# ax1.errorbar(k+0.0015, QSO_mean, yerr = np.array([QSO_low, QSO_high]), fmt='.', label = 'QSO', capsize = 2.0)
# ax1.hlines(3.0364, xmin = 0.135, xmax = 0.285, linestyle = 'dashed')
# ax1.set_ylabel(param)

# param = parameters[1]
# LRG_mean = []
# LRG_low = []
# LRG_high = []
# ELG_mean = []
# ELG_low = []
# ELG_high = []
# QSO_mean = []
# QSO_low = []
# QSO_high = []
# for j in range(8):
#     LRG_mean.append(data_LRG[j][param][1])
#     LRG_low.append(data_LRG[j][param][1] - data_LRG[j][param][0])
#     LRG_high.append(data_LRG[j][param][2] - data_LRG[j][param][1])
#     ELG_mean.append(data_ELG[j][param][1])
#     ELG_low.append(data_ELG[j][param][1] - data_ELG[j][param][0])
#     ELG_high.append(data_ELG[j][param][2] - data_ELG[j][param][1])
#     QSO_mean.append(data_QSO[j][param][1])
#     QSO_low.append(data_QSO[j][param][1] - data_QSO[j][param][0])
#     QSO_high.append(data_QSO[j][param][2] - data_QSO[j][param][1])


# ax2.errorbar(k-0.0015, LRG_mean, yerr = np.array([LRG_low, LRG_high]), fmt='.', label = 'LRG', capsize = 2.0)
# ax2.errorbar(k, ELG_mean, yerr = np.array([ELG_low, ELG_high]), fmt='.', label = 'ELG', capsize = 2.0)
# ax2.errorbar(k+0.0015, QSO_mean, yerr = np.array([QSO_low, QSO_high]), fmt='.', label = 'QSO', capsize = 2.0)
# ax2.hlines(0.6736, xmin = 0.135, xmax = 0.285, linestyle = 'dashed')
# ax2.set_ylabel(param)

# param = parameters[2]
# LRG_mean = []
# LRG_low = []
# LRG_high = []
# ELG_mean = []
# ELG_low = []
# ELG_high = []
# QSO_mean = []
# QSO_low = []
# QSO_high = []
# for j in range(8):
#     LRG_mean.append(data_LRG[j][param][1])
#     LRG_low.append(data_LRG[j][param][1] - data_LRG[j][param][0])
#     LRG_high.append(data_LRG[j][param][2] - data_LRG[j][param][1])
#     ELG_mean.append(data_ELG[j][param][1])
#     ELG_low.append(data_ELG[j][param][1] - data_ELG[j][param][0])
#     ELG_high.append(data_ELG[j][param][2] - data_ELG[j][param][1])
#     QSO_mean.append(data_QSO[j][param][1])
#     QSO_low.append(data_QSO[j][param][1] - data_QSO[j][param][0])
#     QSO_high.append(data_QSO[j][param][2] - data_QSO[j][param][1])


# ax3.errorbar(k-0.0015, LRG_mean, yerr = np.array([LRG_low, LRG_high]), fmt='.', label = 'LRG', capsize = 2.0)
# ax3.errorbar(k, ELG_mean, yerr = np.array([ELG_low, ELG_high]), fmt='.', label = 'ELG', capsize = 2.0)
# ax3.errorbar(k+0.0015, QSO_mean, yerr = np.array([QSO_low, QSO_high]), fmt='.', label = 'QSO', capsize = 2.0)
# ax3.hlines(0.12, xmin = 0.135, xmax = 0.285, linestyle = 'dashed')
# ax3.set_ylabel(param)

# param = parameters[3]
# LRG_mean = []
# LRG_low = []
# LRG_high = []
# ELG_mean = []
# ELG_low = []
# ELG_high = []
# QSO_mean = []
# QSO_low = []
# QSO_high = []
# for j in range(8):
#     LRG_mean.append(data_LRG[j][param][1])
#     LRG_low.append(data_LRG[j][param][1] - data_LRG[j][param][0])
#     LRG_high.append(data_LRG[j][param][2] - data_LRG[j][param][1])
#     ELG_mean.append(data_ELG[j][param][1])
#     ELG_low.append(data_ELG[j][param][1] - data_ELG[j][param][0])
#     ELG_high.append(data_ELG[j][param][2] - data_ELG[j][param][1])
#     QSO_mean.append(data_QSO[j][param][1])
#     QSO_low.append(data_QSO[j][param][1] - data_QSO[j][param][0])
#     QSO_high.append(data_QSO[j][param][2] - data_QSO[j][param][1])


# ax4.errorbar(k-0.0015, LRG_mean, yerr = np.array([LRG_low, LRG_high]), fmt='.', label = 'LRG', capsize = 2.0)
# ax4.errorbar(k, ELG_mean, yerr = np.array([ELG_low, ELG_high]), fmt='.', label = 'ELG', capsize = 2.0)
# ax4.errorbar(k+0.0015, QSO_mean, yerr = np.array([QSO_low, QSO_high]), fmt='.', label = 'QSO', capsize = 2.0)
# ax4.hlines(0.02237, xmin = 0.135, xmax = 0.285, linestyle = 'dashed')
# ax4.set_ylabel(param)
# ax4.set_xlabel(r'$k_{\mathrm{max}} (h/\mathrm{Mpc})$')
# ax4.legend(ncol = 3)

# ##############################                   For figure 13                     ###########################################

# # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True)
# # ax1.errorbar(x_20, x_20**2*xi_data_mono[5:35], yerr = x_20**2*unc_mono_20, fmt = '.', label = 'data', c = 'k', capsize = 1.0, capthick = 0.1, markersize = 1.0, elinewidth = 0.5)
# # ax1.plot(x_20, x_20**2*xi_20_mono, c = 'b', label = r'$s_{min} = 22 h^{-1} Mpc$')
# # ax1.plot(x_30, x_30**2*xi_30_mono, c = 'r', label = r'$s_{min} = 30 h^{-1} Mpc$')
# # ax1.plot(x_40, x_40**2*xi_40_mono, c = 'g', label = r'$s_{min} = 42 h^{-1} Mpc$')
# # ax1.set_ylabel(r'$s^2\xi_0(s)$', labelpad = 0, fontsize = 7)
# # ax1.tick_params(axis='x', labelsize=7, pad = 0)
# # ax1.tick_params(axis='y', labelsize=7, pad = 0)
# # ax2.fill_between(x_20, x_20**2*unc_mono_20, -x_20**2*unc_mono_20, color = 'k', alpha = 0.5, label = 'Single box uncertainty')
# # ax2.plot(x_20, -x_20**2*(xi_data_mono[5:35] - xi_20_mono), c = 'b', label = r'$s_{min} = 22 h^{-1} Mpc$')
# # ax2.plot(x_30, -x_30**2*(xi_data_mono[7:35] - xi_30_mono), c = 'r', label = r'$s_{min} = 30 h^{-1} Mpc$')
# # ax2.plot(x_40, -x_40**2*(xi_data_mono[10:35] - xi_40_mono), c = 'g', label = r'$s_{min} = 42 h^{-1} Mpc$')
# # ax2.set_ylabel(r'$\Delta s^2\xi_0(s)$', labelpad = 0, fontsize = 7)
# # ax2.tick_params(axis='x', labelsize=7, pad = 0)
# # ax2.tick_params(axis='y', labelsize=7, pad = 0)

# # ax3.errorbar(x_20, x_20**2*xi_data_quad[5:35], yerr = x_20**2*unc_quad_20, fmt = '.', label = 'data', c = 'k', capsize = 1.0, capthick = 0.1, markersize = 1.0, elinewidth = 0.5)
# # ax3.plot(x_20, x_20**2*xi_20_quad, c = 'b', label = r'$s_{min} = 22 h^{-1} Mpc$')
# # ax3.plot(x_30, x_30**2*xi_30_quad, c = 'r', label = r'$s_{min} = 30 h^{-1} Mpc$')
# # ax3.plot(x_40, x_40**2*xi_40_quad, c = 'g', label = r'$s_{min} = 42 h^{-1} Mpc$')
# # ax3.set_ylabel(r'$s^2\xi_2(s)$', labelpad = 0, fontsize = 7)
# # ax3.set_xlabel(r's $(h^{-1} Mpc)$', fontsize = 7)
# # ax3.tick_params(axis='x', labelsize=7, pad = 0)
# # ax3.tick_params(axis='y', labelsize=7, pad = 0)
# # ax4.fill_between(x_20, x_20**2*unc_quad_20, -x_20**2*unc_quad_20, color = 'k', alpha = 0.5, label = 'Single box uncertainty')
# # ax4.plot(x_20, -x_20**2*(xi_data_quad[5:35] - xi_20_quad), c = 'b', label = r'$s_{min} = 22 h^{-1} Mpc$')
# # ax4.plot(x_30, -x_30**2*(xi_data_quad[7:35] - xi_30_quad), c = 'r', label = r'$s_{min} = 30 h^{-1} Mpc$')
# # ax4.plot(x_40, -x_40**2*(xi_data_quad[10:35] - xi_40_quad), c = 'g', label = r'$s_{min} = 42 h^{-1} Mpc$')
# # ax4.set_ylabel(r'$\Delta s^2\xi_2(s)$', labelpad = 0, fontsize = 7)
# # ax4.set_xlabel(r's $(h^{-1} Mpc)$', fontsize = 7)
# # ax4.tick_params(axis='x', labelsize=7, pad = 0)
# # ax4.tick_params(axis='y', labelsize=7, pad = 0)
# # ax4.legend(fontsize = 5, ncol = 2)
# # plt.savefig(fname = 'LRG_xi_bestfit.png', dpi=300, bbox_inches = 'tight')