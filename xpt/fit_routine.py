import lsqfit
import numpy as np
import gvar as gv
import sys
import os
# local modules 
import non_analytic_functions as naf
import i_o

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
        self.y = {datatag : self.data['eps_'+datatag] for datatag in self.model_info['particles']}
        #self.y = {datatag : self.data['eps_'+datatag] for datatag in self.model_info['particles']}
        #need to save self.y to generate fit , correlated with self.y
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
    @property
    def extrapolate(self):
        if self._extrapolate is None:
            extrapolate = self.extrapolation
            self._extrapolate = extrapolate
        return self._extrapolate

    @property
    def extrapolation(self):
        extrapolation = Proton(datatag='proton',model_info=self.model_info).extrapolate_mass(observable='sigma_pi_n')
        return extrapolation

    # def extrapolate(self, observable=None, p=None,  data=None):
    #     if p is None:
    #         p = {}
    #         p.update(p)
    #     if data is None:
    #         data = self.pp_data
    #     p.update(data)
    #     output = {}
    #     for lsqfit_model in self._make_models():
    #         obs = lsqfit_model.datatag
    #         output[obs] = lsqfit_model.fitfcn(p)

    #     if observable is not None:
    #         return output[observable]
    #     return output

    def _make_models(self, model_info=None):
        if model_info is None:
            model_info = self.model_info.copy()

        models = np.array([])

        if 'proton' in model_info['particles']:
            models = np.append(models,Proton(datatag='proton', model_info=model_info))

        if 'delta' in model_info['particles']:
            models = np.append(models,Delta(datatag='delta', model_info=model_info))

        if 'sigma_pi_n' in model_info['particles']:
            models = np.append(models,Sigma_pi_N(datatag='sigma_pi_n', model_info=model_info))

        if 'Fpi' in model_info['particles']:
            models = np.append(models,Fpi(datatag='Fpi', model_info=model_info))

 
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
                
                if value == 'llo':
                    orders = ['llo']
                elif value == 'lo':
                    #print('hello')
                    orders = ['llo', 'lo']
                elif value == 'nlo':
                    orders = ['llo', 'lo', 'nlo']
                elif value == 'n2lo':
                    orders = ['llo', 'lo', 'nlo','n2lo']
                elif value == 'n4lo':
                    orders = ['llo', 'lo', 'nlo','n2lo', 'n4lo']
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
            new_prior['eps2_a'] = data['eps2_a']
            new_prior['m_pi'] = data['m_pi']
            new_prior['lam_chi'] = data['lam_chi']
        if self.model_info['order_disc'] is not None:
            #new_prior['eps2_a'] = data['eps2_a']
            #new_prior['m_pi'] = data['m_pi']
            new_prior['lam_chi'] = data['lam_chi']


        # for key in ['eps2_a','lam_chi','m_pi','eps_pi']: 
        #     new_prior[key] = data[key]
        return new_prior

    def _get_prior_keys(self, particle = 'all', order='all',lec_type='all'):
        if particle == 'all':
            output = []
            for particle in ['proton', 'delta','Fpi']:
                keys = self._get_prior_keys(particle=particle, order=order, lec_type=lec_type)
                output.extend(keys)
            return np.unique(output)

        elif order == 'all':
            output = []
            for order in ['llo', 'lo', 'nlo','n2lo','n4lo']:
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
            for p in ['proton', 'delta','Fpi']:
                output[p] = {}
                for o in ['llo', 'lo', 'nlo','n2lo','n4lo']:
                    output[p][o] = {}

            output['proton']['llo' ]['light'  ] = ['m_{proton,0}']
            output['proton']['lo'  ]['disc'   ] = ['d_{proton,a}']
            output['proton']['lo'  ]['light'  ] = ['b_{proton,2}','l4_bar']
            output['proton']['lo'  ]['strange'] = ['d_{proton,s}']
            output['proton']['lo'  ]['xpt']     = ['l4_bar']
            output['proton']['nlo' ]['xpt'    ] = ['g_{proton,proton}', 'g_{proton,delta}','m_{delta,0}']
            output['proton']['n2lo']['disc'   ] = ['d_{proton,aa}', 'd_{proton,al}']
            output['proton']['n2lo']['strange'] = ['d_{proton,as}', 'd_{proton,ls}','d_{proton,ss}']
            output['proton']['n2lo']['light'  ] = ['b_{proton,4}']
            output['proton']['n2lo']['xpt'    ] = ['a_{proton,4}', 'g_{proton,4}']
            #output['proton']['n4lo']['disc'   ] = ['d_{proton,all}', 'd_{proton,aal}']
            output['proton']['n4lo']['strange'] = []
            output['proton']['n4lo']['light'  ] = ['b_{proton,6}']
            output['proton']['n4lo']['xpt'    ] = []

            output['Fpi']['lo']['light']    = ['F0','l4_bar']
            output['Fpi']['lo']['disc']     = ['d_{fpi,a}']
            output['Fpi']['lo']['xpt']      = ['l4_bar']
            output['Fpi']['n2lo']['light']  = ['d_{fpi,ll}','b_{fpi,4}']
            output['Fpi']['n2lo']['disc']   = ['d_{fpi,aa}','d_{fpi,al}']
            output['Fpi']['n2lo']['xpt']    = [ 'a_{fpi,4}','l4_bar']

            output['delta']['llo' ]['light'  ] = ['m_{delta,0}']
            output['delta']['lo'  ]['disc'   ] = ['d_{delta,a}']
            output['delta']['lo'  ]['light'  ] = ['b_{delta,2}']
            output['delta']['lo'  ]['strange'] = ['d_{delta,s}']
            output['delta']['nlo' ]['xpt'    ] = ['g_{delta,delta}', 'g_{proton,delta}', 'm_{proton,0}']
            output['delta']['n2lo']['disc'   ] = ['d_{delta,aa}', 'd_{delta,al}']
            output['delta']['n2lo']['strange'] = ['d_{delta,as}', 'd_{delta,ls}','d_{delta,ss}']
            output['delta']['n2lo']['light'  ] = ['b_{delta,4}']
            output['delta']['n2lo']['xpt'    ] = ['a_{delta,4}', 'g_{delta,4}']

            if lec_type in output[particle][order]:
                return output[particle][order][lec_type]
            else:
                return []

class Proton(lsqfit.MultiFitterModel):
    def __init__(self, datatag, model_info):
        super(Proton, self).__init__(datatag)
        self.model_info = model_info
    
    def fitfcn(self, p): #data=None):
        # if data is not None:
        #     for key in data.keys():
        #         p[key] = data[key] 

        xdata = {}
        #xdata['lam_chi'] = p['lam_chi']
        if self.model_info['fit_phys_units']:
            xdata['eps_pi'] = p['m_pi'] / p['lam_chi']
        elif self.model_info['fit_fpi_units']:
            xdata['eps_pi'] = p['eps_pi']
        
        #
        # xdata['eps_delta'] = (p['m_{delta,0}'] - p['m_{proton,0}']) / p['lam_chi']
        if self.model_info['order_disc'] is not None:
            xdata['eps2_a'] = p['eps2_a']

        if self.model_info['order_strange'] is not None:
            xdata['d_eps2_s'] = (2 *p['m_k']**2 - p['m_pi']**2) / p['lam_chi']**2 - 0.3513

        output = 0
        if self.model_info['sigma']: # fit to obtain nucleon mass derivative for analytic sigma term #
            output += self.fitfcn_lo_sigma(p,xdata)
            output += self.fitfcn_nlo_sigma(p,xdata)
            output += self.fitfcn_n2lo_sigma(p,xdata)
        elif self.model_info['sigma'] is False:
            output += self.fitfcn_llo(p,xdata)
            #output +=  p['m_{proton,0}']
            output += self.fitfcn_lo(p,xdata) 
            output += self.fitfcn_lo_xpt(p,xdata)
            output += self.fitfcn_nlo_xpt(p,xdata) 
            output += self.fitfcn_n2lo(p,xdata)
            output += self.fitfcn_n2lo_xpt(p,xdata)
            output += self.fitfcn_n4lo(p,xdata)
        return output

    def fitfcn_sigma_pi(self, p): 
        xdata = {}
        #xdata['lam_chi'] = p['lam_chi']
        if self.model_info['fit_phys_units']:
            xdata['eps_pi'] = p['m_pi'] / p['lam_chi']
        elif self.model_info['fit_fpi_units']:
            xdata['eps_pi'] = p['eps_pi']
        if self.model_info['order_disc'] is not None:
            xdata['eps2_a'] = p['eps2_a']

        if self.model_info['order_strange'] is not None:
            xdata['d_eps2_s'] = (2 *p['m_k']**2 - p['m_pi']**2) / p['lam_chi']**2 - 0.3513

        output = 0
        #if self.model_info['sigma']: # fit to obtain nucleon mass derivative for analytic sigma term #
        output += self.fitfcn_lo_sigma(p,xdata)
        output += self.fitfcn_nlo_sigma(p,xdata)
        output += self.fitfcn_n2lo_sigma(p,xdata)
        
        return output

    #if self.model_includes[]

    def fitfcn_llo(self,p,xdata):
        output = 0
        if self.model_info['order_light'] in ['llo','lo', 'nlo', 'n2lo','n4lo']:
            output+= p['m_{proton,0}']
        return output

    def extrapolate_mass(self,observable=None,p=None, xdata=None):
        if observable == 'sigma_pi':
            return self.fitfcn_sigma_pi(p)

        if observable == 'proton' :
            return self.fitfcn(p)

    def fitfcn_lo_xpt(self,p,xdata):
        if self.model_info['xpt']:
            output = (p['l4_bar'] + p['b_{proton,2}'])  * (xdata['eps_pi']**2 * p['m_{proton,0}'] * np.log(xdata['eps_pi']**2))
        else:
            return 0
        return output

    def fitfcn_lo(self, p, xdata):
        output = 0
        # if self.model_info.iincludes(lec='light',order='lo')
        if self.model_info['fit_phys_units']: # lam_chi dependence ON #
            if self.model_info['order_disc']    in  ['lo', 'nlo', 'n2lo']:
                output += p['m_{proton,0}'] * (p['d_{proton,a}'] * xdata['eps2_a'])
        
            if self.model_info['order_light'] in ['lo','nlo','n2lo']:
                output+= p['b_{proton,2}'] * xdata['lam_chi'] * xdata['eps_pi']**2 

            if self.model_info['order_strange'] in ['lo', 'nlo', 'n2lo']:
                output+= p['m_{proton,0}']*   (p['d_{proton,s}'] * xdata['d_eps2_s'])

        elif self.model_info['fit_fpi_units']: # lam_chi dependence OFF #
            if self.model_info['order_disc']    in  ['lo', 'nlo', 'n2lo']:
                output += (p['d_{proton,a}'] * xdata['eps2_a'])
            
            if self.model_info['order_light'] in ['lo','nlo','n2lo','n4lo']:
                output+= p['l4_bar'] + p['b_{proton,2}']

            if self.model_info['order_strange'] in ['lo', 'nlo', 'n2lo']:
                output+= (p['d_{proton,s}'] * xdata['d_eps2_s'])

            if self.model_info['order_chiral'] in ['lo', 'nlo', 'n2lo']: #include chiral log
                output += (xdata['eps_pi']**2) * np.log(xdata['eps_pi']**2)

        return output

    def fitfcn_nlo_xpt(self,p,xdata):
        output = 0
        if self.model_info['order_chiral'] in ['nlo','n2lo']:
            if self.model_info['fit_phys_units']: # lam_chi dependence ON #
    
                if self.model_info['delta']:
                    output += -3*np.pi/2 * p['g_{proton,proton}']**2 * xdata['lam_chi'] * xdata['eps_pi']**3 
                    output += -4/3 * p['g_{proton,delta}']**2 * xdata['lam_chi'] * naf.fcn_F(xdata['eps_pi'],xdata['eps_delta'])
                elif self.model_info['delta'] is False:
                    output += -3*np.pi/2 * p['g_{proton,proton}']**2 * xdata['lam_chi'] * xdata['eps_pi']**3
                elif self.model_info['axial'] is False:
                    output += -3*np.pi/2 * xdata['lam_chi'] * xdata['eps_pi']**3

            elif self.model_info['fit_fpi_units']: # lam_chi dependence OFF #
                if self.model_info['delta']: # include F function #
                    output += -3*np.pi/2 * p['g_{proton,proton}']**2 * xdata['eps_pi']**3 
                    output += -4/3 * p['g_{proton,delta}']**2 * naf.fcn_F(xdata['eps_pi'],xdata['eps_delta'])
                elif self.model_info['delta'] is False: # exclude F, J functions
                    output += -3*np.pi/2 * p['g_{proton,proton}']**2 * xdata['eps_pi']**3
                elif self.model_info['axial'] is False:
                    output += -3*np.pi/2 * xdata['eps_pi']**3


        elif self.model_info['xpt'] is False:
            return 0
        
        return output

    def fitfcn_n2lo_xpt(self,p,xdata):
        output = 0
        if self.model_info['order_chiral'] in ['n2lo']:
            if self.model_info['xpt']:
                if self.model_info['fit_phys_units']: 
                    if self.model_info['delta']:
                        output+= (p['g_{proton,4}'] * xdata['lam_chi']* xdata['eps_pi']**2 * naf.fcn_J(xdata['eps_pi'],xdata['eps_delta']))
                    elif self.model_info['delta'] is False:
                        output += p['g_{proton,4}'] * xdata['lam_chi']* xdata['eps_pi']**2

                elif self.model_info['fit_fpi_units']:
                        if self.model_info['delta']:
                            output+= (p['g_{proton,4}'] * xdata['eps_pi']**2 * naf.fcn_J(xdata['eps_pi'],xdata['eps_delta']))
                        elif self.model_info['delta'] is False:
                            output += p['g_{proton,4}'] * xdata['eps_pi']**2
        else:
            return 0
        return output


    def fitfcn_n2lo(self,p,xdata):
        output = 0
        if self.model_info['fit_phys_units']: 
            if self.model_info['order_strange'] in ['n2lo']:  
                output += p['m_{proton,0}']*(
                p['d_{proton,as}']* xdata['eps2_a'] * xdata['d_eps2_s'] +
                (p['d_{proton,ls}'] * xdata['d_eps2_s'] * xdata['eps_pi']**2) + (p['d_{proton,ss}'] * xdata['d_eps2_s']**2)
            )
            if self.model_info['order_disc'] in ['n2lo']:
                output += p['m_{proton,0}']*( 
                (p['d_{proton,al}'] * xdata['eps2_a'] * xdata['eps_pi']**2) 
                + (p['d_{proton,aa}'] * xdata['eps2_a']**2))

            if self.model_info['order_light'] in ['n2lo']:
                output += p['m_{proton,0}'] * (p['b_{proton,4}']*xdata['lam_chi']*xdata['eps_pi']**4)
                
            if self.model_info['order_chiral'] in ['n2lo']:
                output+= p['a_{proton,4}']*xdata['lam_chi'] * xdata['eps_pi']**4 * np.log(xdata['eps_pi']**2)

        elif self.model_info['fit_fpi_units']:
            if self.model_info['order_strange'] in ['n2lo','n4lo']:  
                output += p['m_{proton,0}']*(
                p['d_{proton,as}']* xdata['eps2_a'] * xdata['d_eps2_s'] +
                p['d_{proton,ls}'] * xdata['d_eps2_s'] * xdata['eps_pi']**2 + p['d_{proton,ss}'] * xdata['d_eps2_s']**2
            )
            if self.model_info['order_disc'] in ['n2lo','n4lo']:
                output += p['m_{proton,0}']*( 
                (p['d_{proton,al}'] * xdata['eps2_a'] * xdata['eps_pi']**2) 
                + (p['d_{proton,aa}'] * xdata['eps2_a']**2))

            if self.model_info['order_light'] in ['n2lo','n4lo']:
                output += p['b_{proton,4}']*xdata['eps_pi']**4
                
            if self.model_info['order_chiral'] in ['n2lo','n4lo']:
                output+= p['a_{proton,4}'] * xdata['eps_pi']**4 * np.log(xdata['eps_pi']**2)

    
        return output

    def fitfcn_n4lo(self,p,xdata):
        output = 0
        if self.model_info['fit_phys_units']:
            if self.model_info['order_light'] in ['n4lo']:
                output += xdata['lam_chi'] * (
                + xdata['eps_pi']**6 *p['b_{proton,6}'])
            if self.model_info['order_disc'] in ['n4lo']:
                output += xdata['lam_chi'] * (
                + p['d_{proton,all}'] * xdata['eps2_a'] * xdata['eps_pi']**4
                + p['d_{proton,aal}'] * xdata['eps2_a']**2 * xdata['eps_pi']**2
                + p['d_{proton,aal}'] * xdata['eps2_a']**3)

        elif self.model_info['fit_fpi_units']:
            if self.model_info['order_light'] in ['n4lo']:
                output += xdata['eps_pi']**6 *p['b_{proton,6}'] + (xdata['eps_pi']**4 * xdata['eps2_a']**2) 
            if self.model_info['order_disc'] in ['n4lo']:
                output += (
                + p['d_{proton,all}'] * xdata['eps2_a'] * xdata['eps_pi']**4
                + p['d_{proton,aal}'] * xdata['eps2_a']**2 * xdata['eps_pi']**2
                + p['d_{proton,aal}'] * xdata['eps2_a']**3)

        return output

# expansion of nucleon mass derivative for term in analytic sigma term #

    def fitfcn_lo_sigma(self, p, xdata):
        output = 0
        if self.model_info['order_disc'] in ['lo', 'nlo', 'n2lo']:
            output += p['m_{proton,0}'] * (p['d_{proton,a}'] * xdata['eps2_a'])
    
        if self.model_info['order_light'] in ['lo', 'nlo', 'n2lo']:
                output+= p['b_{proton,2}'] *xdata['eps_pi']* (2*xdata['lam_chi']*xdata['eps_pi'])

        if self.model_info['order_strange'] in ['lo', 'nlo', 'n2lo']:
            output+= p['m_{proton,0}']*(p['d_{proton,s}'] *  xdata['d_eps2_s'])

        return output

    def fitfcn_nlo_sigma(self,p,xdata):
        output = 0
        if self.model_info['xpt'] is True:
            output += -3*np.pi/2 * p['g_{proton,proton}']**2 * xdata['eps_pi'] * (3*xdata['lam_chi'] *xdata['eps_pi']**2) 
            output -= 4/3 * p['g_{proton,delta}']**2 * xdata['eps_pi'] * (xdata['lam_chi']*naf.fcn_dF(xdata['eps_pi'],xdata['eps_delta']))

        if self.model_info['xpt'] is False:
            return 0
        
        return output

    def fitfcn_n2lo_sigma(self,p,xdata):
        output = 0
        if self.model_info['order_strange'] in ['n2lo']:  
            output += p['m_{proton,0}']*(
            p['d_{proton,as}']* xdata['eps2_a'] * xdata['d_eps2_s'] +
            (p['d_{proton,ls}'] * xdata['d_eps2_s'] * xdata['eps_pi']**2) + (p['d_{proton,ss}'] * xdata['d_eps2_s']**2)
        )
        if self.model_info['order_disc'] in ['n2lo']:
            output += p['m_{proton,0}']*( 
            (p['d_{proton,al}'] * xdata['eps2_a'] * xdata['eps_pi']**2) 
            + (p['d_{proton,aa}'] * xdata['eps2_a']**2))

        if self.model_info['order_light'] in ['n2lo']:
            output += p['m_{proton,0}'] * (p['b_{proton,4}']* xdata['eps_pi'] * (4* xdata['lam_chi']*xdata['eps_pi']**3))
            
        # if self.model_info['order_chiral'] in ['n2lo']:
        #     output+= p['g_{proton,4}'] * xdata['eps_pi']* (2*xdata['eps_pi'] * xdata['lam_chi'] * naf.fcn_dJ(xdata['eps_pi'],xdata['eps_delta'])
        #     + xdata['lam_chi']*xdata['eps_pi']**2* naf.fcn_dJ(xdata['eps_pi'],xdata['eps_delta']))
        #     + p['a_{proton,4}']* xdata['eps_pi']* (4* xdata['lam_chi'] * xdata['eps_pi']**3 * np.log(xdata['eps_pi']**2 + 2* xdata['lam_chi']*xdata['eps_pi']**3))

        if self.model_info['order_chiral'] in ['n2lo']:
            output+= p['g_{proton,4}'] * xdata['eps_pi']* (2*xdata['eps_pi'] * xdata['lam_chi'])
            + xdata['lam_chi']*xdata['eps_pi']**2
            + p['a_{proton,4}']* xdata['eps_pi']* (4* xdata['lam_chi'] * xdata['eps_pi']**3 * np.log(xdata['eps_pi']**2 + 2* xdata['lam_chi']*xdata['eps_pi']**3))
    
        return output

    def buildprior(self, prior, mopt=False, extend=False):
        return prior

    def builddata(self, data):
        return data[self.datatag]

class Sigma_pi_N(lsqfit.MultiFitterModel):
    def __init__(self, datatag, model_info):
        super(Sigma_pi_N, self).__init__(datatag)
        self.model_info = model_info
    
    def fitfcn(self, p): #data=None):
        
        xdata = {}
        #xdata['lam_chi'] = p['lam_chi']
        if self.model_info['fit_phys_units']:
            xdata['eps_pi'] = p['m_pi'] / p['lam_chi']
        elif self.model_info['fit_fpi_units']:
            xdata['eps_pi'] = p['eps_pi']
        
        #
        # xdata['eps_delta'] = (p['m_{delta,0}'] - p['m_{proton,0}']) / p['lam_chi']
        if self.model_info['order_disc'] is not None:
            xdata['eps2_a'] = p['eps2_a']

        if self.model_info['order_strange'] is not None:
            xdata['d_eps2_s'] = (2 *p['m_k']**2 - p['m_pi']**2) / p['lam_chi']**2 - 0.3513

        output = 0
        #if self.model_info['sigma']: # fit to obtain nucleon mass derivative for analytic sigma term #
        output += self.fitfcn_lo_sigma(p,xdata)
        output += self.fitfcn_nlo_sigma(p,xdata)
        output += self.fitfcn_n2lo_sigma(p,xdata)
        
        return output

    def fitfcn_lo_sigma(self, p, xdata):
        output = 0
        if self.model_info['order_disc'] in ['lo', 'nlo', 'n2lo']:
            output += p['m_{proton,0}'] * (p['d_{proton,a}'] * xdata['eps2_a'])
    
        if self.model_info['order_light'] in ['lo', 'nlo', 'n2lo']:
                output+= p['b_{proton,2}'] *xdata['eps_pi']* (2*xdata['lam_chi']*xdata['eps_pi'])

        if self.model_info['order_strange'] in ['lo', 'nlo', 'n2lo']:
            output+= p['m_{proton,0}']*(p['d_{proton,s}'] *  xdata['d_eps2_s'])

        return output

    def fitfcn_nlo_sigma(self,p,xdata):
        output = 0
        if self.model_info['xpt'] is True:
            output += -3*np.pi/2 * p['g_{proton,proton}']**2 * xdata['eps_pi'] * (3*xdata['lam_chi'] *xdata['eps_pi']**2) 
            output -= 4/3 * p['g_{proton,delta}']**2 * xdata['eps_pi'] * (xdata['lam_chi']*naf.fcn_dF(xdata['eps_pi'],xdata['eps_delta']))

        if self.model_info['xpt'] is False:
            return 0
        
        return output

    def fitfcn_n2lo_sigma(self,p,xdata):
        output = 0
        if self.model_info['order_strange'] in ['n2lo']:  
            output += p['m_{proton,0}']*(
            p['d_{proton,as}']* xdata['eps2_a'] * xdata['d_eps2_s'] +
            (p['d_{proton,ls}'] * xdata['d_eps2_s'] * xdata['eps_pi']**2) + (p['d_{proton,ss}'] * xdata['d_eps2_s']**2)
        )
        if self.model_info['order_disc'] in ['n2lo']:
            output += p['m_{proton,0}']*( 
            (p['d_{proton,al}'] * xdata['eps2_a'] * xdata['eps_pi']**2) 
            + (p['d_{proton,aa}'] * xdata['eps2_a']**2))

        if self.model_info['order_light'] in ['n2lo']:
            output += p['m_{proton,0}'] * (p['b_{proton,4}']*   (4* xdata['lam_chi']*xdata['eps_pi']**3))
            
        # if self.model_info['order_chiral'] in ['n2lo']:
        #     output+= p['g_{proton,4}'] * xdata['eps_pi']* (2*xdata['eps_pi'] * xdata['lam_chi'] * naf.fcn_dJ(xdata['eps_pi'],xdata['eps_delta'])
        #     + xdata['lam_chi']*xdata['eps_pi']**2* naf.fcn_dJ(xdata['eps_pi'],xdata['eps_delta']))
        #     + p['a_{proton,4}']* xdata['eps_pi']* (4* xdata['lam_chi'] * xdata['eps_pi']**3 * np.log(xdata['eps_pi']**2 + 2* xdata['lam_chi']*xdata['eps_pi']**3))

        if self.model_info['order_chiral'] in ['n2lo']:
            output+= p['g_{proton,4}'] * xdata['eps_pi']* (2*xdata['eps_pi'] * xdata['lam_chi'])
            + xdata['lam_chi']*xdata['eps_pi']**2
            + p['a_{proton,4}']* xdata['eps_pi']* (4* xdata['lam_chi'] * xdata['eps_pi']**3 * np.log(xdata['eps_pi']**2 + 2* xdata['lam_chi']*xdata['eps_pi']**3))
    
        return output

    def buildprior(self, prior, mopt=False, extend=False):
        return prior

    def builddata(self, data):
        return data[self.datatag]

## have to fit LO, n2lo for FPI ## 

class Fpi(lsqfit.MultiFitterModel):
    def __init__(self, datatag, model_info):
        super(Fpi, self).__init__(datatag)
        self.model_info = model_info
    
    def fitfcn(self, p): #data=None):
        xdata = {}
        #xdata['lam_chi'] = p['lam_chi']
        if self.model_info['fit_phys_units']:
            xdata['eps_pi'] = p['m_pi'] / p['lam_chi']
        elif self.model_info['fit_fpi_units']:
            xdata['eps_pi'] = p['eps_pi']
        # xdata['eps_delta'] = (p['m_{delta,0}'] - p['m_{proton,0}']) / p['lam_chi']
        if self.model_info['order_disc'] is not None:
            xdata['eps2_a'] = p['eps2_a']

        if self.model_info['order_strange'] is not None:
            xdata['d_eps2_s'] = (2 *p['m_k']**2 - p['m_pi']**2) / p['lam_chi']**2 - 0.3513

        # USING BARYON POWER COUNTING FOR ORDERING #
        output = 0
        output +=  p['F0']
        output += self.fitfcn_lo(p,xdata) 
        output += self.fitfcn_nlo_xpt(p,xdata)
        output += self.fitfcn_n2lo(p,xdata)
        output += self.fitfcn_n2lo_xpt(p,xdata)
        
        return output

    #F_pi = F_0 ( 1 + eps_pi^2 ( -log(eps_pi^2) + l_4 ) )

    def fitfcn_lo(self,p,xdata):
        output = 0
        if self.model_info['order_light'] in ['lo','n2lo']:
            if self.model_info['xpt_fpi']:
                output+= p['F0']*xdata['eps_pi']**2*(p['l4_bar']) * (-np.log(xdata['eps_pi']**2))
            else:
                output +=p['F0']*xdata['eps_pi']**2*(p['l4_bar'])

        if self.model_info['order_disc'] in ['lo','n2lo']:
            output += p['F0'] * (p['d_{fpi,a}']  * xdata['eps_pi']**2)

        return output

    def fitfcn_nlo_xpt(self,p,xdata):
        output = 0
        if self.model_info['xpt_fpi']:
            output -= 1/4*p['F0']*xdata['eps_pi']**4*(np.log(xdata['eps_pi']**2))
        else:
            return 0
        return output

    def fitfcn_n2lo(self,p,xdata):
        output = 0
    
        if self.model_info['order_light'] in ['n2lo']:
            output += p['F0']*(p['d_{fpi,ll}'] *p['b_{fpi,4}']* xdata['eps_pi']**2)

        if self.model_info['order_disc'] in ['n2lo']:
            output += p['F0']*( 
            (p['d_{fpi,al}'] * xdata['eps2_a'] * xdata['eps_pi']**2) 
            + (p['d_{fpi,aa}'] * xdata['eps2_a']**2))
        return output

    def fitfcn_n2lo_xpt(self,p,xdata):
        output = 0

        if self.model_info['xpt_fpi']:
            output += xdata['eps_pi']**4 * np.log(xdata['eps_pi']**2)*(-p['F0']*p['a_{fpi,4}'] - 2*p['F0']*p['l4_bar'])
            - xdata['eps_pi']**4*(p['F0'])

        else:
            return 0

    def buildprior(self, prior, mopt=False, extend=False):
        return prior

    def builddata(self, data):
        return data[self.datatag]

class Delta(lsqfit.MultiFitterModel):
    def __init__(self, datatag, model_info):
        super(Delta, self).__init__(datatag)
        self.model_info = model_info
    
    def fitfcn(self, p, data=None):
        '''
        mass of the ith delta in chiral expansion:
        \Delta = quark mass independent delta-nucleon mass splitting
        M_0(\Delta) = renormalized nucleon mass in chiral limit 

        M_B_i = 
        M_0(\Delta) - M_B_i^1(\mu,\Delta) - M_B_i^3/2(\mu,\Delta) - M_B_i^2(\mu,\Delta) + ...
        '''
        if data is not None:
            for key in data.keys():
                p[key] = data[key] 

        xdata = {}
        xdata['lam_chi'] = p['lam_chi']
        #xdata['eps_pi'] = p['m_pi'] / p['lam_chi']
        #xdata['eps_delta'] = (p['m_{delta,0}'] - p['m_{proton,0}']) / p['lam_chi']
        xdata['eps2_a'] = p['eps2_a']
        xdata['d_eps2_s'] = (2 *p['m_k']**2 - p['m_pi']**2) / p['lam_chi']**2 - 0.3513

        output = 0
        
        output += self.fitfcn_lo(p,xdata) 
        output += self.fitfcn_nlo(p,xdata) 
        output += self.fitfcn_n2lo(p,xdata)
        output += self.fitfcn_n4lo(p,xdata)
        if self.model_info['lam_chi'] is True:
            output = output / xdata['lam_chi']
        output  +=  p['m_{delta,0}']
        return output

    def fitfcn_lo(self, p, xdata):
        output = 0
        if self.model_info['order_disc'] in ['lo', 'nlo', 'n2lo']:
            output += p['m_{proton,0}'] * (p['d_{delta,a}'] * xdata['eps2_a'])
    
        if self.model_info['order_light'] in ['lo', 'nlo', 'n2lo']:
                output+= p['b_{delta,2}'] * xdata['lam_chi'] * xdata['eps_pi']**2

        if self.model_info['order_strange'] in ['lo', 'nlo', 'n2lo']:
            output+= p['m_{delta,0}']*(p['d_{delta,s}'] *  xdata['d_eps2_s'])

        return output

    def fitfcn_nlo(self,p,xdata):
        output = 0
        if self.model_info['xpt'] is True:
            output += (
            -25*np.pi/54 * p['g_{delta,delta}']**2 * xdata['lam_chi'] * xdata['eps_pi']**3 
            -1/3 * p['g_{proton,delta}']**2 * xdata['lam_chi'] * naf.fcn_F(xdata['eps_pi'],-xdata['eps_delta'])
            )    
        if self.model_info['xpt'] is False:
            return 0
        
        return output

    def fitfcn_n2lo(self,p,xdata):
        output = 0
        if self.model_info['order_strange'] in ['n2lo']:  
            output += p['m_{delta,0}']*(
            p['d_{delta,as}']* xdata['eps2_a'] * xdata['d_eps2_s'] +
            (p['d_{delta,ls}'] * xdata['d_eps2_s'] * xdata['eps_pi']**2) + (p['d_{delta,ss}'] * xdata['d_eps2_s']**2)
        )
        if self.model_info['order_disc'] in ['n2lo']:
            output += p['m_{delta,0}']*( 
            (p['d_{delta,al}'] * xdata['eps2_a'] * xdata['eps_pi']**2) 
            + (p['d_{delta,aa}'] * xdata['eps2_a']**2)
            )

        if self.model_info['order_light'] in ['n2lo']:
            output += p['m_{delta,0}'] * (p['b_{delta,4}']*xdata['lam_chi']*xdata['eps_pi']**4)
            
        if self.model_info['order_chiral'] in ['n2lo']:
            output+= (p['g_{delta,4}'] * xdata['lam_chi']* xdata['eps_pi']**2 * naf.fcn_J(xdata['eps_pi'],-xdata['eps_delta']))
            + p['a_{delta,4}']*xdata['lam_chi'] * xdata['eps_pi']**4 * np.log(xdata['eps_pi']**2)
    
        return output

    def fitfcn_n4lo(self,p,xdata):
        output = 0
        #if self.model_info['order_light'] in ['n4lo']:
        output += xdata['lam_chi'] * (
            + xdata['eps_pi']**6 *p['b_{delta,6}']
            + p['d_{delta,all}'] * xdata['eps2_a'] * xdata['eps_pi']**4
            + p['d_{delta,aal}'] * xdata['eps2_a']**2 * xdata['eps_pi']**2
            + p['d_{delta,aal}'] * xdata['eps2_a']**3)
        return output

   
    def buildprior(self, prior, mopt=False, extend=False):
        return prior

    def builddata(self, data):
        return data[self.datatag]

# class Sigma_pi_N(lsqfit.MultiFitterModel):
#     def __init__(self, datatag, model_info):
#         super(Sigma_pi_N, self).__init__(datatag)
#         self.model_info = model_info
    
#     def fitfcn(self, p, data=None):
#         '''
#         expansion of the nucleon mass derivative in order to compute the analytical expression for 
#         the sigma_pi_N term. 
#         eps_pi(\partial M_N/ \partial eps_pi)
#         The term d eps_pi * lam_chi that arises in expansion ->0 since of course both scalars

#         '''
#         if data is not None:
#             for key in data.keys():
#                 p[key] = data[key] 

#         xdata = {}
#         xdata['lam_chi'] = p['lam_chi']
#         xdata['eps_pi'] = p['m_pi'] / p['lam_chi']
#         xdata['eps_delta'] = (p['m_{delta,0}'] - p['m_{proton,0}']) / p['lam_chi']
#         xdata['eps2_a'] = p['eps2_a']
#         xdata['d_eps2_s'] = (2 *p['m_k']**2 - p['m_pi']**2) / p['lam_chi']**2 - 0.3513

#         output = 0
#         output += self.fitfcn_lo(p,xdata) 
#         output += self.fitfcn_nlo(p,xdata) 
#         output += self.fitfcn_n2lo(p,xdata)
#         if self.model_info['lam_chi'] is True:
#             output = output / xdata['lam_chi']
#         return output

#     def fitfcn_lo(self, p, xdata):
#         output = 0
#         if self.model_info['order_disc'] in ['lo', 'nlo', 'n2lo']:
#             output += p['m_{proton,0}'] * (p['d_{proton,a}'] * xdata['eps2_a'])
    
#         if self.model_info['order_light'] in ['lo', 'nlo', 'n2lo']:
#                 output+= p['b_{proton,2}'] *xdata['eps_pi']* (2*xdata['lam_chi']*xdata['eps_pi'])

#         if self.model_info['order_strange'] in ['lo', 'nlo', 'n2lo']:
#             output+= p['m_{proton,0}']*(p['d_{proton,s}'] *  xdata['d_eps2_s'])

#         return output

#     def fitfcn_nlo(self,p,xdata):
#         output = 0
#         if self.model_info['xpt'] is True:
#             output += -3*np.pi/2 * p['g_{proton,proton}']**2 * xdata['eps_pi'] * (3*xdata['lam_chi'] *xdata['eps_pi']**2) 
#             output -= 4/3 * p['g_{proton,delta}']**2 * xdata['eps_pi'] * (xdata['lam_chi']*naf.fcn_dF(xdata['eps_pi'],xdata['eps_delta']))


#         if self.model_info['xpt'] is False:
#             return 0
        
#         return output

#     def fitfcn_n2lo(self,p,xdata):
#         output = 0
#         if self.model_info['order_strange'] in ['n2lo']:  
#             output += p['m_{proton,0}']*(
#             p['d_{proton,as}']* xdata['eps2_a'] * xdata['d_eps2_s'] +
#             (p['d_{proton,ls}'] * xdata['d_eps2_s'] * xdata['eps_pi']**2) + (p['d_{proton,ss}'] * xdata['d_eps2_s']**2)
#         )
#         if self.model_info['order_disc'] in ['n2lo']:
#             output += p['m_{proton,0}']*( 
#             (p['d_{proton,al}'] * xdata['eps2_a'] * xdata['eps_pi']**2) 
#             + (p['d_{proton,aa}'] * xdata['eps2_a']**2))

#         if self.model_info['order_light'] in ['n2lo']:
#             output += p['m_{proton,0}'] * (p['b_{proton,4}']* xdata['eps_pi'] * (4* xdata['lam_chi']*xdata['eps_pi']**3))
            
#         if self.model_info['order_chiral'] in ['n2lo']:
#             output+= p['g_{proton,4}'] * xdata['eps_pi']* (2*xdata['eps_pi'] * xdata['lam_chi'] * naf.fcn_dJ(xdata['eps_pi'],xdata['eps_delta'])
#             + xdata['lam_chi']*xdata['eps_pi']**2* naf.fcn_dJ(xdata['eps_pi'],xdata['eps_delta']))
#             + p['a_{proton,4}']* xdata['eps_pi']* (4* xdata['lam_chi'] * xdata['eps_pi']**3 * np.log(xdata['eps_pi']**2 + 2* xdata['lam_chi']*xdata['eps_pi']**3))
    
#         return output

#     # def fitfcn_n4lo(self,p,xdata):
#     #     output = 0
#     #     if self.model_info['order_light'] in ['n4lo']:
#     #         output += xdata['lam_chi'] * (
#     #         + xdata['eps_pi']**6 *p['b_{proton,6}'])
#     #     if self.model_info['order_disc'] in ['n4lo']:
#     #         output += xdata['lam_chi'] * (
#     #         + p['d_{proton,all}'] * xdata['eps2_a'] * xdata['eps_pi']**4
#     #         + p['d_{proton,aal}'] * xdata['eps2_a']**2 * xdata['eps_pi']**2
#     #         + p['d_{proton,aal}'] * xdata['eps2_a']**3)
#     #     return output

#     def buildprior(self, prior, mopt=False, extend=False):
#         return prior

#     def builddata(self, data):
#         return data[self.datatag]









