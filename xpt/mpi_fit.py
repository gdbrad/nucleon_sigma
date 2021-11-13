import lsqfit
import numpy as np
import gvar as gv
import sys
import os
# local modules 
import non_analytic_functions as naf
from i_o import InputOutput

class fit_routine(object):

    def __init__(self, prior, data, model_info):
         
        self.prior = prior
        self.data = data
        #self.load = i_o.InputOutput()
        self.model_info = model_info.copy()
        self._fit = None
        self._extrapolate = None
        self._simultaneous = False
        self._posterior = None
        #self.phys_point_data = loader.get_data_phys_point(param=None)
        mpisq_mev = (self.data['m_pi'] / gv.gvar('0.08730(70)') * 197.3269804)**2
        self.y = {'mpi' : gv.gvar([(g).mean for g in mpisq_mev], [(g).sdev for g in mpisq_mev])} 
        #self.y = {'mpi' self.data['m_pi']**2}
    def __str__(self):
        return str(self.fit)
    
    @property
    def fit(self):
        if self._fit is None:
            models = self._make_models()
            prior = self._make_prior()
            data = self.y
            fitter = lsqfit.MultiFitter(models=models)
            fit = fitter.lsqfit(data=data, prior=prior, fast=False, mopt=False)
            self._fit = fit

        return self._fit 

    def _make_models(self, model_info=None):
        if model_info is None:
            model_info = self.model_info.copy()

        models = np.array([])

        if 'mpi' in model_info['particles']:
            models = np.append(models,Mpi(datatag='mpi', model_info=model_info))
        return models

    def _make_prior(self, data=None):
        '''
        Only need priors for LECs/data needed in fit
        '''
        if data is None:
            data = self.data
        prior = self.prior
        new_prior = {}
        particles = []
        particles.extend(self.model_info['particles'])

        keys = []
        orders = []
        for p in particles:
            for l, value in [('light',self.model_info['order_light']), ('disc', self.model_info['order_disc']),
            ('strange', self.model_info['order_strange']), ('xpt', self.model_info['order_chiral'])]:
            # include all orders equal to and less than desired order in expansion #
                if value == 'lo':
                    orders = ['lo']
                elif value == 'n2lo':
                    orders = ['lo','n2lo']
                # elif value == 'n4lo':
                #     orders = ['llo', 'lo', 'nlo','n2lo', 'n4lo']
                else:
                    orders = []
                for o in orders:
                    keys.extend(self._get_prior_keys(particle=p, order=o, lec_type = l))
                
        for k in keys:
            new_prior[k] = prior[k]

        if self.model_info['order_strange'] is not None:
            new_prior['m_k'] = data['m_k']
            #new_prior['eps_pi'] = data['eps_pi']
        if self.model_info['order_light'] is not None:
            new_prior['eps_pi'] = data['eps_pi']
            new_prior['m_q'] = data['m_q']
            #new_prior['B'] = data['B']
            #new_prior['m_pi_sq'] = data['m_pi_sq']
            #new_prior['eps2_a'] = data['eps2_a']
            new_prior['m_pi'] = data['m_pi']
            #new_prior['lam_chi'] = data['lam_chi']
        if self.model_info['order_disc'] is not None:
            new_prior['eps2_a'] = data['eps2_a']
            #new_prior['m_pi'] = data['m_pi']
            #new_prior['lam_chi'] = data['lam_chi']

        for key in ['eps_pi']:
            new_prior[key] = data[key]
        # for key in ['eps2_a','lam_chi','m_pi','eps_pi']: 
        #     new_prior[key] = data[key]
        return new_prior

    def _get_prior_keys(self, particle = 'all', order='all',lec_type='all'):
        if particle == 'all':
            output = []
            for particle in ['mpi']:
                keys = self._get_prior_keys(particle=particle, order=order, lec_type=lec_type)
                output.extend(keys)
            return np.unique(output)

        elif order == 'all':
            output = []
            for order in ['llo', 'lo', 'nlo','n2lo']:
                keys = self._get_prior_keys(particle=particle, order=order, lec_type=lec_type)
                output.extend(keys)
            return np.unique(output)

        elif lec_type == 'all':
            output = []
            for lec_type in ['disc','light','strange','xpt']:
                keys = self._get_prior_keys(particle=particle, order=order, lec_type=lec_type)
                output.extend(keys)
            return np.unique(output)

        else: 
            # construct dict of lec names corresponding to particle, order, lec_type #
            output = {}
            for p in ['mpi']:
                output[p] = {}
                for o in ['llo', 'lo', 'nlo','n2lo']:
                    output[p][o] = {}
            
            output['mpi']['lo']['light']    = ['B','l_3']
            output['mpi']['lo']['disc']     = ['d_{mpi,a}']
            output['mpi']['lo']['xpt']      = []
            output['mpi']['n2lo']['light']  = ['c_{mpi,2F}']
            output['mpi']['n2lo']['disc']   = ['d_{mpi,aa}','d_{mpi,al}']
            output['mpi']['n2lo']['xpt']    = ['c_{mpi,1F}']

            if lec_type in output[particle][order]:
                return output[particle][order][lec_type]
            else:
                return []

class Mpi(lsqfit.MultiFitterModel):
    def __init__(self, datatag, model_info):
        super(Mpi, self).__init__(datatag)
        self.model_info = model_info
        self.phys = InputOutput().get_data_phys_point()
    
    def fitfcn(self, p,xdata=None):
        xdata = {}
        #xdata['lam_chi'] = p['lam_chi']
        if self.model_info['fit_phys_units']:
            xdata['eps_pi'] = p['m_pi'] / p['lam_chi']
        elif self.model_info['fit_fpi_units']:
            xdata['eps_pi'] = p['eps_pi']
        if self.model_info['order_disc'] is not None:
            xdata['eps2_a'] = p['eps2_a']

        # M0 = M^2 = 2mB
        msq = 2 * p['B'] * p['m_q'] 
        output = msq *(
        + 1
        + self.fitfcn_lo(p,xdata) 
        + self.fitfcn_lo_xpt(p,xdata)
        + self.fitfcn_n2lo(p,xdata)
        + self.fitfcn_n2lo_xpt(p,xdata)
        )
        return output

    def fitfcn_lo(self,p,xdata):

        output = 0
        if self.model_info['order_light'] in ['lo','n2lo']:
            output += xdata['eps_pi']**2 * (2*p['l_3'])
    
        if self.model_info['order_disc'] in ['lo','n2lo']:
            output += (p['d_{mpi,a}'] * xdata['eps2_a'])
        return output

    def fitfcn_lo_xpt(self, p,xdata):
        output = 0
        if self.model_info['order_chiral'] in ['lo','n2lo']:
            output +=  xdata['eps_pi']**2 * (1/2 * np.log(xdata['eps_pi']**2))

        return output

    def fitfcn_n2lo_xpt(self,p,xdata):
        output = 0
        if self.model_info['order_chiral'] in ['n2lo']:
            output += 7/8*(xdata['eps_pi']**4*(np.log(xdata['eps_pi']**2)**2 +p['c_{mpi,1F}']*np.log(xdata['eps_pi']**2)))
             
        return output

    def fitfcn_n2lo(self,p,xdata):
        output = 0
        if self.model_info['order_light'] in ['n2lo']:
            output += xdata['eps_pi']**4*(p['c_{mpi,2F}'])

        if self.model_info['order_disc'] in ['n2lo']:
            output +=  p['d_{mpi,al}'] * xdata['eps2_a'] * xdata['eps_pi']**4 +p['d_{mpi,aa}'] * xdata['eps2_a']**2

        return output

    def buildprior(self, prior, mopt=False, extend=False):
        return prior

    def builddata(self, data):
        return data[self.datatag]
