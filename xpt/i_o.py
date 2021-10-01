import numpy as np
import gvar as gv
import sys
import datetime
import re
import os
#import yaml
import h5py

class InputOutput(object):
    def __init__(self):
        project_path = os.path.normpath(os.path.join(os.path.realpath(__file__), os.pardir, os.pardir))

        with h5py.File(project_path+'/data/hyperon_data.h5', 'r') as f:
            ens_hyp = sorted(list(f.keys()))
            ens_hyp = sorted([e.replace('_hp', '') for e in  ens_hyp])

        with h5py.File(project_path+'/data/input_data.h5', 'r') as f: 
            ens_in = sorted(list(f.keys()))

        ensembles = sorted(list(set(ens_hyp) & set(ens_in)))
        ensembles.remove('a12m220')
        ensembles.remove('a12m220ms')
        ensembles.remove('a12m310XL')
        ensembles.remove('a12m220S')
        ensembles.remove('a12m180L')
        ensembles.remove('a15m310')


        self.ensembles = ensembles
        self.project_path = project_path

    def _get_bs_data(self, scheme=None,units=None):
        to_gvar = lambda arr : gv.gvar(arr[0], arr[1])
        #hbar_c = self.get_data_phys_point('hbarc') # MeV-fm (PDG 2019 conversion constant)
        scale_factors = gv.load(self.project_path +'/data/scale_setting.p')

        data = {}
        with h5py.File(self.project_path+'/data/input_data.h5', 'r') as f: 
            for ens in self.ensembles:
                data[ens] = {}
                if scheme in ['w0_org','w0_imp'] and units=='phys':
                    data[ens]['units'] = hbar_c *scale_factors[scheme+':'+ens[:3]] /scale_factors[scheme+':w0']
                elif scheme in ['w0_org', 'w0_imp'] and units=='Fpi':
                    data[ens]['units'] = scale_factors[scheme+':'+ens[:3]]
                #data[ens]['units_MeV'] = hbar_c / to_gvar(f[ens]['a_fm'][scheme][:])
                data[ens]['alpha_s'] = f[ens]['alpha_s']
                data[ens]['L'] = f[ens]['L']
                data[ens]['m_pi'] = f[ens]['mpi'][1:]
                data[ens]['m_k'] = f[ens]['mk'][1:]
                data[ens]['lam_chi'] = 4 *np.pi *f[ens]['Fpi'][1:]
                data[ens]['eps_Fpi'] = f[ens]['Fpi'][1:]
                

                
                # if units=='Fpi':
                #     data[ens]['units'] = 1/data[ens]['lam_chi'] #for removing lam_chi dependence of fits 
                data[ens]['eps_pi'] = data[ens]['m_pi'] / data[ens]['lam_chi']
                data[ens]['eps_k'] = data[ens]['m_k']/data[ens]['lam_chi']
                data[ens]['eps2_a'] = (1 / (2 *to_gvar(f[ens]['w0a_callat_imp']))**2) 


        with h5py.File(self.project_path+'/data/hyperon_data.h5', 'r') as f:
            for ens in self.ensembles:
                for obs in list(f[ens]):
                    data[ens].update({obs : f[ens][obs][:]})
                if ens+'_hp' in list(f):
                    for obs in list(f[ens+'_hp']):
                        data[ens].update({obs : f[ens+'_hp'][obs][:]})

                for obs in ['proton','delta']:
                    data[ens]['eps_'+obs] = data[ens]['m_'+obs] / data[ens]['lam_chi']
                    #data[ens]['eps_'+obs] = data[ens]['m_'+obs]
        return data

    def get_data(self, scheme=None,units='Fpi',include_phys=None):
        bs_data = self._get_bs_data(scheme,units)
        phys_data = self.get_data_phys_point(param='m_proton')

        gv_data = {}
        
        dim1_obs = ['m_k','m_pi','eps_pi','lam_chi','m_proton','m_delta','eps_proton','eps_Fpi']
        for ens in self.ensembles:
            gv_data[ens] = {}
            for obs in dim1_obs:
                gv_data[ens][obs] = bs_data[ens][obs] #- np.mean(bs_data[ens][obs]) + bs_data[ens][obs][0]

            gv_data[ens] = gv.dataset.avg_data(gv_data[ens], bstrap=True) 
            gv_data[ens]['eps2_a'] = bs_data[ens]['eps2_a']

        ensembles = list(gv_data)
        # include physical pt as data pt to (potentially) anchor the fit #
        if include_phys:
            data_pp = self.get_data_phys_point()
            gv_data['a00m135'] = {}
            gv_data['a00m135']['eps_proton'] = data_pp['eps_proton']
            gv_data['a00m135']['eps_delta'] = data_pp['eps_delta']
            gv_data['a00m135']['eps_pi'] = data_pp['eps_pi']
            gv_data['a00m135']['eps2_a'] = gv.gvar(0, 10e-8)
            gv_data['a00m135']['eps_k'] = data_pp['eps_k']

            ensembles.insert(0, 'a00m135')
        output = {}
        for param in gv_data[self.ensembles[0]]:
            output[param] = np.array([gv_data[ens][param] for ens in self.ensembles])
        return output, ensembles


    def get_data_phys_point(self, param=None):
        data_pp = {
            'eps2_a' : gv.gvar(0),
            'a' : gv.gvar(0),
            'alpha_s' : gv.gvar(0.0),
            'L' : gv.gvar(np.infty),
            'hbarc' : gv.gvar(197.3269804, 0), # MeV-fm

            'lam_chi' : 4 *np.pi *gv.gvar('92.07(57)'),
            'm_pi' : gv.gvar('134.8(3)'), # '138.05638(37)'
            'm_k' : gv.gvar('494.2(3)'), # '495.6479(92)'
            'eps_pi' : gv.gvar('134.8(3)') / (4 *np.pi *gv.gvar('92.07(57)')),
            'eps_k' : gv.gvar('494.2(3)') / (4 *np.pi *gv.gvar('92.07(57)')),
            'm_proton' : np.mean(gv.gvar(['938.272081(06)', '939.565413(06)'])),
            'eps_proton' : np.mean(gv.gvar(['938.272081(06)', '939.565413(06)'])) /  (4 *np.pi *gv.gvar('92.07(57)')) ,
            'm_delta' : gv.gvar(1232, 2),
            'eps_delta' : gv.gvar(1232, 2) / (4 *np.pi *gv.gvar('92.07(57)')),
            # 'm_lambda' : gv.gvar(1115.683, 0.006),
            # 'm_sigma' : np.mean(gv.gvar(['1189.37(07)', '1192.642(24)', '1197.449(30)'])),
            # 'm_sigma_st' : np.mean(gv.gvar(['1382.80(35)', '1383.7(1.0)', '1387.2(0.5)'])),
            # 'm_xi' : np.mean(gv.gvar(['1314.86(20)', '1321.71(07)'])),
            # 'm_xi_st' : np.mean(gv.gvar(['1531.80(32)', '1535.0(0.6)'])),
            # 'm_omega' : gv.gvar(1672.45, 0.29)
        }
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
        for key in ['eps_pi']:#,'lam_chi', ]:
            new_prior[key] = data[key] #/data['lam_chi']
        return new_prior




