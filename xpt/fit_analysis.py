import lsqfit
import numpy as np
import gvar as gv
import time
import matplotlib
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d.axes3d import Axes3D
import os
import h5py
import yaml
import sys
sys.setrecursionlimit(10000)

import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['figure.figsize']  = [6.75, 6.75/1.618034333]
mpl.rcParams['font.size']  = 20
mpl.rcParams['legend.fontsize'] =  16
mpl.rcParams["lines.markersize"] = 5
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['text.usetex'] = True


#internal xpt modules
import fit_routine as fit
import i_o

class fit_analysis(object):
    
    def __init__(self, phys_point_data, data=None, model_info=None, prior=None):
        project_path = os.path.normpath(os.path.join(os.path.realpath(__file__), os.pardir, os.pardir))

        with h5py.File(project_path+'/data/hyperon_data.h5', 'r') as f:
            ens_hyp = sorted(list(f.keys()))
            ens_hyp = sorted([e.replace('_hp', '') for e in  ens_hyp])

        with h5py.File(project_path+'/data/input_data.h5', 'r') as f: 
            ens_in = sorted(list(f.keys()))

        ensembles = sorted(list(set(ens_hyp) & set(ens_in)))
        ensembles.remove('a12m220')
        ensembles.remove('a12m220S')
        #data,ensembles = i_o.InputOutput.get_data(scheme='w0_imp')

        self.ensembles = ensembles
        self.model_info = model_info
        self.data = data
        self.fitter = {}
        self._input_prior = prior
        self._phys_point_data = phys_point_data
        self._fit = {}
        self.fitter = fit.fit_routine(prior=prior,data=data, model_info=model_info)

        # def __str__(self):
        #     output = "Model: %s" %(self.model) 
        #     output += '\nError Budget:\n'
        #     max_len = np.max([len(key) for key in self.error_budget[obs]])
        #     for key in {k: v for k, v in sorted(self.error_budget[obs].items(), key=lambda item: item[1], reverse=True)}:
        #         output += '  '
        #         output += key.ljust(max_len+1)
        #         output += '{: .1%}\n'.format((self.error_budget[obs][key]/self..sdev)**2).rjust(7)

            
        #     return output

    ## It is particularly useful for analyzing the impact of the a priori uncertainties encoded in the prior
    @property
    def error_budget(self):
        output = ''
    
        output += '\n'
        output += str(self._get_error_budget(particle='proton'))
        return output

    def _get_error_budget(self, **kwargs):
        
        output = None
        

        #use above dict to fill in values where particle name goes.. leave hardcoded for now
        # strange_keys = [
        # 'd_{lambda,s}','d_{sigma,s}', 'd_{sigma_st,s}', 'd_{xi,s}', 'd_{xi_st,s}',
        # 'd_{lambda,as}', 'd_{lambda,ls}', 'd_{lambda,ss}', 'd_{sigma,as}', 'd_{sigma,ls}', 'd_{sigma,ss}',
        # 'd_{sigma_st,as}', 'd_{sigma_st,ls}', 'd_{sigma_st,ss}', 'd_{xi,as}', 'd_{xi,ls}', 'd_{xi,ss}',
        # 'd_{xi_st,as}', 'd_{xi_st,ls}', 'd_{xi_st,ss}']
        
        # chiral_keys = [
        # 's_{lambda}', 's_{sigma}', 's_{sigma,bar}', 's_{xi}', 's_{xi,bar}', 
        # 'g_{lambda,sigma}', 'g_{lambda,sigma_st}', 'g_{sigma,sigma}', 'g_{sigma_st,sigma}', 
        # 'g_{sigma_st,sigma_st}', 'g_{xi,xi}', 'g_{xi_st,xi}', 'g_{xi_st,xi_st}', 'b_{lambda,4}', 
        # 'b_{sigma,4}', 'b_{sigma_st,4}', 'b_{xi,4}', 'b_{xi_st,4}', 'a_{lambda,4}', 'a_{sigma,4}', 
        # 'a_{sigma_st,4}', 'a_{xi,4}', 'a_{xi_st,4}'] 
        
        # disc_keys = [
        # 'm_{lambda,0}', 'm_{sigma,0}', 'm_{sigma_st,0}', 'm_{xi,0}', 'm_{xi_st,0}', 'd_{lambda,a}', 'd_{sigma,a}',  
        # 'd_{sigma_st,a}', 'd_{xi,a}',  'd_{xi_st,a}', 'd_{lambda,aa}', 'd_{lambda,al}', 
        # 'd_{sigma,aa}', 'd_{sigma,al}',  'd_{sigma_st,aa}', 'd_{sigma_st,al}', 
        # 'd_{xi,aa}', 'd_{xi,al}',  'd_{xi_st,aa}', 'd_{xi_st,al}']

        strange_keys = [
            'd_{proton,s}','d_{proton,as}', 'd_{proton,ls}','d_{proton,ss}'
        ]

        chiral_keys = [
            'g_{proton,proton}', 'g_{proton,delta}','m_{delta,0}', 'a_{proton,4}', 'g_{proton,4}'
        ]

        disc_keys = [
            'd_{proton,a}', 'd_{proton,aa}', 'd_{proton,al}', 'd_{proton,all}', 'd_{proton,aal}'
        ]

        light_keys = [
            'm_{proton,0}', 'b_{proton,2}', 'b_{proton,4}', 'b_{proton,6}'
        ]
        
        phys_keys = list(self.phys_point_data)
        stat_keys = ['lam_chi','eps2_a','m_lambda','m_pi','m_k']
        
        mdls = fit.fit_routine(prior=self.prior, data=self.data, model_info=self.model_info)

        # if verbose:
        #     if output is None:
        #         output = ''

        result = {}
        result['disc'] = mdls.fit.p.partialsdev(
            [self.prior[key] for key in disc_keys if key in self.prior]
        )
        result['chiral'] = mdls.fit.p.partialsdev(
            [self.prior[key] for key in chiral_keys if key in self.prior]
        )
        result['strange'] = mdls.fit.p.partialsdev(
            [self.prior[key] for key in strange_keys if key in self.prior]
        )
        result['light'] = mdls.fit.p.partialsdev(
            [self.prior[key] for key in light_keys if key in self.prior]
        )
        result['pp_input'] = mdls.fit.p.partialsdev(
            [self.phys_point_data[key] for key in phys_keys]
        )
        # output['stat'] = value.partialsdev(
        #     [self._get_prior(stat_keys), self.fitter.y]
        #     #self.fitter['w0'].y
        # )
        return result

        # # xpt/chiral contributions
        # inputs.update({str(param)+' [disc]' : self._input_prior[param] for param in disc_keys if param in self._input_prior})
        # inputs.update({str(param)+' [xpt]' : self._input_prior[param] for param in chiral_keys if param in self._input_prior})
        # inputs.update({str(param)+ '[strange]' : self._input_prior[param] for param in strange_keys if param in self._input_prior})

        # # phys point contributions
        # inputs.update({str(param)+' [pp]' : self.phys_point_data[param] for param in list(phys_keys)})

        inputs.update({str(param): self._input_prior[param] for param in disc_keys if param in self._input_prior})
        inputs.update({str(param): self._input_prior[param] for param in chiral_keys if param in self._input_prior})
        inputs.update({str(param): self._input_prior[param] for param in strange_keys if param in self._input_prior})

        # phys point contributions
        inputs.update({str(param): self.phys_point_data[param] for param in list(phys_keys)})
        #del inputs['lam_chi [pp]']

        #stat contribtions
        inputs.update({'x [stat]' : self._input_prior[param] for param in stat_keys if param in self._input_prior})# , 'y [stat]' : self.fitter.fit.y})
        print(inputs.values())

        if kwargs is None:
            kwargs = {}
        kwargs.setdefault('percent', False)
        kwargs.setdefault('ndecimal', 10)
        kwargs.setdefault('verify', True)
        
        #output = {}

        #output = {}
        extrapolated_mass = {}
        for particle in self.model_info['particles']:
            extrapolated_mass[particle] = self.fitfcn(p=self.posterior, data=self.phys_point_data, particle=particle)
        #print(inputs.keys())
        #print(extrapolated_mass)

        #value = extrapolated_mass.partialsdev([self.prior[key] for key in disc_keys if key in self.prior])
        #for keys in disc_keys:
        print(gv.fmt_errorbudget(outputs=extrapolated_mass, inputs=inputs, verify=True))
        output = {}

        output['disc'] = extrapolated_mass[particle].partialsdev(
                    [self.prior[particle] for key in disc_keys if key in self.prior])
        output['pp'] = extrapolated_mass[particle].partialsdev(
                    [self.prior for key in phys_keys if key in self.prior])

        output['stat'] = extrapolated_mass[particle].partialsdev(
                    [self.prior for key in stat_keys if key in self.prior])

        output['chiral'] = extrapolated_mass[particle].partialsdev(
                    [self.prior for key in chiral_keys if key in self.prior])
        
        
        print(output)
        #         #elif observable == 't0':
        # #output += 'observable: ' + observable + '\n' + gv.fmt_errorbudget(outputs={'t0' : self.sqrt_t0}, inputs=inputs, **kwargs) + '\n---\n'

        #print(value)

        # output['disc'] = value.partialsdev([self.prior[key] for key in disc_keys if key in self.prior]
        # )
        # output['chiral'] = value.partialsdev([self.prior[key] for key in chiral_keys if key in self.prior]
        # )
        # output['strange'] = value.partialsdev([self.prior[key] for key in strange_keys if key in self.prior]
        # )
        # output['pp_input'] = value.partialsdev([self.phys_point_data[key] for key in phys_keys]
        # )
        # output['stat'] = value.partialsdev([self.prior[key] for key in stat_keys if key in self.prior]
        # )

        

        # #     #output += '\n' + gv.fmt_errorbudget(outputs=outputs, inputs=inputs, **kwargs) + '\n---\n'
        # #     # elif== 't0':
        # #     #     output +=  ' ++ '\n' + gv.fmt_errorbudget(outputs={'t0' : self.sqrt_t0}, inputs=inputs, **kwargs) + '\n---\n'



        #return output

    # @property
    # def fit_info(self):
    #     #fit_info = {}
    #     fit_info = {
    #         'name' : self.model,
    #         #'w0_imp' : self.w0,
    #         'logGBF' : self.fitter.logGBF,
    #         'chi2/df' : self.fitter.chi2 / self.fitter.dof,
    #         'Q' : self.fit.Q,
    #         'phys_point' : self.phys_point_data,
    #         #'error_budget' : self.error_budget['w0'],
    #         'prior' : self.prior,
    #         'posterior' : self.posterior
    #     }
    #     return fit_info

    # Returns names of LECs in prior/posterior
    @property
    def extrapolated_mass(self):
        extrapolated_mass = {}
        for particle in self.model_info['particles']:
            extrapolated_mass[particle] = self.fitfcn(p=self.posterior, data=self.phys_point_data, particle=particle)
        print(extrapolated_mass)
        return extrapolated_mass


    @property
    def fit_keys(self):
        output = {}
        
        keys1 = list(self._input_prior.keys())
        keys2 = list(self.fitter.fit.p.keys())
        output = np.intersect1d(keys1, keys2)
        return output

    @property
    def model(self):
        return self.model_info['name']

    @property
    def phys_point_data(self):
        return self._get_phys_point_data()

    # need to convert to/from lattice units
    def _get_phys_point_data(self, parameter=None):
        if parameter is None:
            return self._phys_point_data
        else:
            return self._phys_point_data[parameter]

    @property
    def posterior(self):
        return self._get_posterior()

    # # Returns dictionary with keys fit parameters, entries gvar results

    def _get_posterior(self,param=None):
        #output = {}
        #return self.fit.p
        if param == 'all':
            return self.fitter.fit.p
        elif param is not None:
            return self.fitter.fit.p[param]
        else:
            output = {}
            for param in self._input_prior:
                if param in self.fitter.fit.p:
                    output[param] = self.fitter.fit.p[param]
            return output

    @property
    def prior(self):
        return self._get_prior()

    def _get_prior(self, param=None):
        output = {}
        if param is None:
            output = {param : self.fitter.fit.prior}
        elif param == 'all':
            output = self.fitter.fit.prior
        else:
            output = self.fitter.fit.prior[param]

        return output

    @property
    def hyp_mass(self):
        return self.fitfcn(data=self.phys_point_data.copy())

    def _extrapolate_to_ens(self, ens=None, phys_params=None):
        if phys_params is None:
            phys_params = []

        extrapolated_values = {}
        for j, ens_j in enumerate(self.ensembles):
            posterior = {}
            xdata = {}
            if ens is None or (ens is not None and ens_j == ens):
                for param in self.fitter.fit.p:
                    shape = self.fitter.fit.p[param].shape
                    if param in phys_params:
                        posterior[param] = self.phys_point_data[param] / self.phys_point_data['hbarc']
                    elif shape == ():
                        posterior[param] = self.fitter.fit.p[param]
                    else:
                        posterior[param] = self.fitter.fit.p[param][j]

                if 'alpha_s' in phys_params:
                    posterior['alpha_s'] = self.phys_point_data['alpha_s']

                if 'eps_pi' in phys_params:
                    xdata['eps_pi'] = self.phys_point_data['m_pi'] / self.phys_point_data['lam_chi']
                if 'd_eps2_s' in phys_params:
                    xdata['d_eps2_s'] = (2 *self.phys_point_data['m_k']**2 - self.phys_point_data['m_pi']**2)/ self.phys_point_data['lam_chi']**2
                if 'eps_a' in phys_params:
                    xdata['eps_a'] = 0

                if ens is not None:
                    return self.fitfcn(p=posterior, data={}, particle=None)
                else:
                    extrapolated_values[ens_j] = self.fitfcn(p=posterior, data={}, particle=None)

                
            extrapolated_values[ens_j] = self.fitfcn(p=posterior, data={}, particle=None)
        return extrapolated_values

    def fitfcn(self, p, data=None, particle=None):
        output = {}
        # if p is None:
        #     p = self.posterior

        for mdl in self.fitter._make_models():
            part = mdl.datatag
            output[part] = mdl.fitfcn(p,data)

        if particle is None:
            return output
        else:
            return output[particle]



    def shift_latt_to_phys(self, ens=None, phys_params=None):
        value_shifted = {}
        for j, ens_j in enumerate(self.ensembles):
            if ens is None or ens_j == ens:
                value_latt = self.fit.y.values()[0][j]
                value_fit = self._extrapolate_to_ens(ens=j)
                value_fit_phys = self._extrapolate_to_ens(ens_j, phys_params)

                value_shifted[ens_j] = value_latt + value_fit_phys - value_fit
                if ens is not None:
                    return value_shifted[ens_j]

        return value_shifted