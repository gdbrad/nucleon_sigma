import numpy as np
import gvar as gv
from pathlib import Path
import sys
import datetime
import re
import os
#import yaml
import h5py
import priors

def get_data_and_prior(units,system,scheme,convert_data=None,decorr_scale=None):
    prior = priors.get_prior(units=units)
    input_output = InputOutput(units=units, scheme=scheme, system=system, convert_data=convert_data,decorr_scale=decorr_scale)
    
    data = input_output.perform_gvar_processing()
    new_prior = input_output.make_prior(data=data, prior=prior)
    
    if units == 'fpi':
        phys_point_data = input_output.get_data_phys_point(units='fpi')
    else:
        phys_point_data = input_output.get_data_phys_point(units='phys')
    
    return data, new_prior, phys_point_data


class InputOutput:
    '''Bootstrapped data ingestion and output to gvar average datasets'''
    def __init__(self,
                 scheme:str,
                 units:str,
                 system:str,
                 convert_data=None, # convert data to fpi or phys units at time of ingestion 
                 decorr_scale=None, # decorrelate lattice spacing between a06,a09 etc.
                 decorr_scale_full=None, # decorrelate lattice spacing between all individual ensembles
                ):
        
        self.scheme = scheme # Valid choices for scheme: 't0_org', 't0_imp', 'w0_org', 'w0_imp'
        self.units = units # physical or fpi units
        self.system = system # proton/delta or just single
        self.convert_data = convert_data
        self.decorr_scale = decorr_scale
        self.decorr_scale_full = decorr_scale_full
        cwd = Path(os.getcwd())
        project_root = cwd.parent
        self.data_dir = os.path.join(project_root, "data")
        data_path_hyperon = os.path.join(self.data_dir, "hyperon_data.h5")
        data_path_input = os.path.join(self.data_dir, "input_data.h5")

        # bootstrapped hyperon correlator data
        with h5py.File(data_path_hyperon, 'r') as f:
            ens_hyp = sorted(list(f.keys()))
            ens_hyp = sorted([e.replace('_hp', '') for e in  ens_hyp])

        # bootstrapped scale setting data 
        with h5py.File(data_path_input, 'r') as f: 
            ens_in = sorted(list(f.keys()))

        ensembles = sorted(list(set(ens_hyp) & set(ens_in)))
        # hopefully with new data these can be safely re-added
        ensembles.remove('a12m220')
        ensembles.remove('a12m220ms')
        ensembles.remove('a12m310XL')
        ensembles.remove('a12m220S')
        ensembles.remove('a12m180L')
        self.ensembles = ensembles
        if self.system == 'all':
            self.dim1_obs = ['m_proton', 'm_delta','m_pi','m_k','lam_chi','eps_pi']
        if self.system == 'proton':
            self.dim1_obs=['m_proton','m_pi','m_k','lam_chi','eps_pi']
        if self.system == 'delta':
            self.dim1_obs=['m_delta', 'm_pi','m_k','lam_chi','eps_pi']
        if self.system == 'fpi':
            self.dim1_obs=['Fpi','m_pi','m_k','lam_chi','eps_pi']


    def _get_bs_data(self, scheme=None,units='phys'):
        to_gvar = lambda arr : gv.gvar(arr[0], arr[1])
        hbar_c = self.get_data_phys_point('hbarc') # MeV-fm (PDG 2019 conversion constant)
        scale_factors = gv.load(self.data_dir +'/scale_setting.p')
        # a_fm =  gv.load(self.project_path +'/a_fm_results.p')
        if scheme is None:
            scheme = 'w0_imp'
        if scheme not in ['t0_org', 't0_imp', 'w0_org', 'w0_imp']:
            raise ValueError('Invalid scale setting scheme')

        data = {}
        with h5py.File(self.data_dir+'/input_data.h5', 'r') as f: 
            for ens in self.ensembles:
                data[ens] = {}
                if scheme in ['w0_org','w0_imp'] and units=='phys':
                    data[ens]['units'] = hbar_c *scale_factors[scheme+':'+ens[:3]] /scale_factors[scheme+':w0']
                elif scheme in ['w0_org', 'w0_imp'] and units=='Fpi':
                    data[ens]['units'] = scale_factors[scheme+':'+ens[:3]]
                data[ens]['units_MeV'] = hbar_c / to_gvar(f[ens]['a_fm'][scheme][:])
                data[ens]['alpha_s'] = f[ens]['alpha_s']
                data[ens]['L'] = f[ens]['L'][()]
                data[ens]['m_pi'] = f[ens]['mpi'][1:]
                data[ens]['m_k'] = f[ens]['mk'][1:]
                data[ens]['lam_chi'] = 4 *np.pi *f[ens]['Fpi'][1:]
                data[ens]['Fpi'] = f[ens]['Fpi'][:] 
                data[ens]['eps_pi'] = data[ens]['m_pi'] / data[ens]['lam_chi']
                data[ens]['units_Fpi'] = 1/data[ens]['lam_chi'][:]
                if self.units=='Fpi':
                #     #for data[ens]['units'] not in data[ens]['lam_chi']:
                    data[ens]['units'] =  1/data[ens]['lam_chi'] #for removing lam_chi dependence of fits    
                if scheme == 'w0_imp':
                    data[ens]['eps2_a'] = 1 / (2 *to_gvar(f[ens]['w0a_callat_imp']))**2
                elif scheme ==  'w0_org':
                    data[ens]['eps2_a'] = 1 / (2 *to_gvar(f[ens]['w0a_callat']))**2
                elif scheme == 't0_imp':
                    data[ens]['eps2_a'] = 1 / (4 *to_gvar(f[ens]['t0aSq_imp']))
                elif scheme == 't0_org':
                    data[ens]['eps2_a'] = 1 / (4 *to_gvar(f[ens]['t0aSq']))

        with h5py.File(self.data_dir+'/hyperon_data.h5', 'r') as f:
            for ens in self.ensembles:
                for obs in list(f[ens]):
                    data[ens].update({obs: f[ens][obs][:]})
                if ens+'_hp' in list(f):
                    for obs in list(f[ens+'_hp']):
                        data[ens].update({obs : f[ens+'_hp'][obs][:]})
                if self.units == 'fpi':
                    for obs in ['proton', 'delta']:
                        data[ens].update({'m_'+obs: data[ens]['m_'+obs] / data[ens]['lam_chi'][:]})
        # abstract this to a dict with keys ens #

        # data['a09m135']['m_q']   = ((0.00152 + (0.938 / 10**4)) /  gv.gvar('0.08730(70)'))* hbar_c 
        # data['a09m220']['m_q']   = ((0.00449 + (1.659 / 10**4)) /  gv.gvar('0.08730(70)')) * hbar_c
        # data['a09m310']['m_q']   = ((0.00951 + (2.694 / 10**4)) /  gv.gvar('0.08730(70)')) * hbar_c
        # data['a09m350']['m_q']   = ((0.0121 + (2.560 / 10**4)) /  gv.gvar('0.08730(70)')) * hbar_c
        # data['a09m400']['m_q']   = ((0.0160 + (2.532 / 10**4)) /  gv.gvar('0.08730(70)')) * hbar_c
        # data['a12m130'] ['m_q']  = ((0.00195 + (1.642 / 10**4)) /  gv.gvar('0.12066(88)')) * hbar_c 
        # data['a12m220']['m_q']   = ((0.006   + (4.050 / 10**4)) /  gv.gvar('0.12066(88)')) * hbar_c
        # data['a12m310']['m_q']   = ((0.0126   + (7.702 / 10**4)) /  gv.gvar('0.12066(88)')) * hbar_c 
        # data['a12m350']['m_q']   = ((0.0166   + (7.579 / 10**4)) /  gv.gvar('0.12066(88)')) * hbar_c 
        # data['a12m400']['m_q']   = ((0.0219   + (7.337 / 10**4)) /  gv.gvar('0.12066(88)')) * hbar_c  
        # data['a15m135XL']['m_q'] = ((0.00237 + (2.706 / 10**4)) /  gv.gvar('0.1505(10)')) * hbar_c 
        # data['a15m220']['m_q']   = ((0.00712 + (5.736 / 10**4)) /  gv.gvar('0.1505(10)')) * hbar_c 
        # data['a15m310']['m_q']   = ((0.0158 + (9.563 / 10**4)) /  gv.gvar('0.1505(10)')) * hbar_c 
        # data['a15m350']['m_q']   = ((0.0206 + (9.416 / 10**4))/  gv.gvar('0.1505(10)')) * hbar_c 
        # data['a15m400']['m_q']   = ((0.0278 + (9.365 / 10**4)) /  gv.gvar('0.1505(10)')) * hbar_c
        return data
    
    def perform_gvar_processing(self):
        """convert raw baryon and pseudoscalar data to gvar datasets"""
        bs_data = self._get_bs_data()
        gv_data = {}
        for ens in self.ensembles:
            gv_data[ens] = gv.BufferDict()
            for obs in self.dim1_obs:
                #recentering data to avoid bias inherent from bootstrapping. getting estimate for central value.
                gv_data[ens][obs] = bs_data[ens][obs] - np.mean(bs_data[ens][obs]) + bs_data[ens][obs][0]
                # gv_data[ens][obs] = bs_data[ens][obs] 

            gv_data[ens] = gv.dataset.avg_data(gv_data[ens], bstrap=True) 
            for obs in self.dim1_obs:
            # if self.convert_data:
                if self.units == 'phys':
                        gv_data[ens][obs] = gv_data[ens][obs] *bs_data[ens]['units_MeV']
                else:
                    # if self.units == 'fpi':
                    gv_data[ens][obs] = gv_data[ens][obs]

            gv_data[ens]['eps2_a'] = bs_data[ens]['eps2_a']
            # gv_data[ens]['Fpi'] = bs_data[ens]['Fpi']

            gv_data[ens]['L'] = gv.gvar(bs_data[ens]['L'], bs_data[ens]['L'] / 10**6)
        output = {}
        for param in gv_data[self.ensembles[0]]:
            output[param] = np.array([gv_data[ens][param] for ens in self.ensembles])

        # # include physical pt as data pt to (potentially) anchor the fit #
        # if include_phys:
        #     data_pp = self.get_data_phys_point()
        #     gv_data['a00m135'] = {}
        #     gv_data['a00m135']['eps_proton'] = data_pp['eps_proton']
        #     gv_data['a00m135']['eps_delta'] = data_pp['eps_delta']
        #     gv_data['a00m135']['eps_pi'] = data_pp['eps_pi']
        #     gv_data['a00m135']['eps2_a'] = gv.gvar(0, 10e-8)
        #     gv_data['a00m135']['eps_k'] = data_pp['eps_k']

        #     ensembles.insert(0, 'a00m135')
        return output


    def get_data_phys_point(self, param=None,units=None):
        data_pp = {
            'eps2_a' : gv.gvar(0),
            'a' : gv.gvar(0),
            'alpha_s' : gv.gvar(0.0),
            'L' : gv.gvar(np.infty),
            'hbarc' : 197.3269804, # MeV-fm

            'lam_chi' : 4 *np.pi *gv.gvar('92.07(57)'),
            'm_pi' : gv.gvar('134.8(3)'), # '138.05638(37)'
            'm_k' : gv.gvar('494.2(3)'), # '495.6479(92)'
            'eps_pi' : gv.gvar('134.8(3)') / (4 *np.pi *gv.gvar('92.07(57)')),
            'eps_k' : gv.gvar('494.2(3)') / (4 *np.pi *gv.gvar('92.07(57)')),
            'm_proton' : np.mean(gv.gvar(['938.272081(06)', '939.565413(06)'])),
            'eps_proton' : np.mean(gv.gvar(['938.272081(06)', '939.565413(06)'])) /  (4 *np.pi *gv.gvar('92.07(57)')) ,
            'm_delta' : gv.gvar(1232, 2),
            'eps_delta' : gv.gvar(1232, 2) / (4 *np.pi *gv.gvar('92.07(57)')),
        }
        if units == 'fpi':
            data_pp['m_proton'] = data_pp['eps_proton']
            data_pp['m_delta'] = data_pp['eps_delta']

        if param is not None:
            return data_pp[param]
        
        return data_pp

    @property
    def posterior(self):
        return self.get_posterior()

    def get_posterior(self,fit_test=None,prior=None,param=None):
        if param == 'all':
            return fit_test.p
        elif param is not None:
            return fit_test.p[param]
        else:
            output = {}
            for param in prior:
                if param in fit_test.p:
                    output[param] = fit_test.p[param]
            return output

    def extrapolate(self, fit_test=None,observable=None, p=None,  data=None):
        if p is None:
            p = {}
            p.update(self.posterior)
        if data is None:
            data = self.get_data_phys_point()
        p.update(data)
        output = {}
        for lsqfit_model in fit_test._make_models():
            obs = lsqfit_model.datatag
            output[obs] = lsqfit_model.fitfcn(p)

        if observable is not None:
            return output[observable]
        return output

    def make_prior(self,data,prior):
        new_prior = {}
        for key in prior:
            new_prior[key] = prior[key]
        for key in ['m_pi', 'm_k', 'lam_chi', 'eps2_a']:
            new_prior[key] = data[key]
        return new_prior




