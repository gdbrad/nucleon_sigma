import lsqfit
import numpy as np
import gvar as gv
import copy
import functools
# local modules 
import xpt.non_analytic_functions as naf
import xpt.i_o
import xpt.fpi_fit

class fit_routine():
    '''
    Base lsqfit fitting class for the nucleon and delta mass extrapolations in SU(2) hbxpt
    '''

    def __init__(self, prior, data, model_info,phys_point_data):
         
        self.prior = prior
        self.data = data
        self._phys_point_data = phys_point_data
        self.model_info = model_info.copy()
        self._extrapolate = None
        self._simultaneous = False
        self._posterior = None
        self.y = {datatag : self.data['eps_'+datatag] for datatag in self.model_info['particles']}
    
    def __str__(self):
        return str(self.fit)
    
    @functools.cached_property
    def fit(self):
        models = self._make_models()
        prior = self._make_prior()
        data = self.y
        fitter = lsqfit.MultiFitter(models=models)
        fit = fitter.lsqfit(data=data, prior=prior, fast=False, mopt=False)

        return fit
    
    @property
    def posterior(self):
        return self._get_posterior()

    def _get_posterior(self,param=None):
        if param == 'all':
            return self.fit.p
        if param is not None:
            return self.fit.p[param]
        
        output = {}
        for param in self.prior:
            if param in self.fit.p:
                output[param] = self.fit.p[param]
        return output  
    
    def get_fitfcn(self,p=None,data=None,particle=None,xdata=None):
        output = {}
        if p is None:
            p = copy.deepcopy(self.posterior)
        if data is None:
            data = copy.deepcopy(self.phys_point_data)
        p.update(data)
        for mdl in self._make_models(model_info=self.model_info):
            part = mdl.datatag
            output[part] = mdl.fitfcn(p=p,data=data,xdata=xdata)
        if particle is None:
            return output
        
        return output[particle]
    
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
    def extrapolate(self):
        if self._extrapolate is None:
            extrapolate = self.extrapolation
            self._extrapolate = extrapolate
        return self._extrapolate

    @property
    def extrapolation(self):
        extrapolation = Proton(datatag='proton',model_info=self.model_info)
        extrap = extrapolation.extrapolate_mass(observable='proton')
        return extrap

    # def extrapolate(self, observable=None, p=None,  data=None,c2m=1):
    #     if observable == 'sigma_pi':
    #         if p is None:
    #             p = {}
    #             p.update(self._posterior)
    #         if data is None:
    #             data = self.pp_data
    #         p.update(data)
    #         p_default = {
    #             'l3' : -1/4 * (gv.gvar('3.53(26)') + np.log(self.pp_data['eps_pi']**2))
    #             #'l4' : 
    #         }

    #         return Proton(fcn_sigma_pi(p=p, model_info=self.model_info)

    #     elif observable == 'm_p':
    #         return super().extrapolate(observable = 'eps_p', p=p, data=data)* self.pp_data


    def _make_models(self, model_info=None):
        if model_info is None:
            model_info = self.model_info.copy()

        models = np.array([])

        if 'proton' in model_info['particles']:
            models = np.append(models,Proton(datatag='proton', model_info=model_info))

        if 'delta' in model_info['particles']:
            models = np.append(models,Delta(datatag='delta', model_info=model_info))

        # if 'sigma_pi_n' in model_info['particles']:
        #     models = np.append(models,Sigma_pi_N(datatag='proton', model_info=model_info))

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
                    keys.extend(self._get_prior_keys(particle=p, order=o, lec_type=l))
                
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
            new_prior['lam_chi'] = data['lam_chi']

        return new_prior

    def _get_prior_keys(self, particle = 'all', order='all',lec_type='all'):
    
        if particle == 'all':
            output = []
            for particle in ['proton', 'delta']:
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
        # ensure the following output initialization and indentation is correct or else an annoying key error will arise !!#
        else: 
            # construct dict of lec names corresponding to particle, order, lec_type #
            output = {}
            for p in ['proton','delta']:
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
            output['proton']['n4lo']['disc'   ] = ['d_{proton,all}', 'd_{proton,aal}']
            output['proton']['n4lo']['strange'] = []
            output['proton']['n4lo']['light'  ] = ['b_{proton,6}']
            output['proton']['n4lo']['xpt'    ] = []

            output['delta']['llo' ]['light'  ] = ['m_{delta,0}']
            output['delta']['lo'  ]['disc'   ] = ['d_{delta,a}']
            output['delta']['lo'  ]['light'  ] = ['b_{delta,2}']
            output['delta']['lo'  ]['strange'] = ['d_{delta,s}']
            output['delta']['nlo' ]['xpt'    ] = ['g_{delta,delta}', 'g_{proton,delta}', 'm_{proton,0}']
            output['delta']['n2lo']['disc'   ] = ['d_{delta,aa}', 'd_{delta,al}']
            output['delta']['n2lo']['strange'] = ['d_{delta,as}', 'd_{delta,ls}','d_{delta,ss}']
            output['delta']['n2lo']['light'  ] = ['b_{delta,4}']
            output['delta']['n2lo']['xpt'    ] = ['a_{delta,4}', 'g_{delta,4}']
            output['delta']['n4lo']['strange'] = []
            output['delta']['n4lo']['light'  ] = ['b_{delta,6}']
            output['delta']['n4lo']['xpt'    ] = []

            if lec_type in output[particle][order]:
                return output[particle][order][lec_type]
            else:
                return []

class Proton(lsqfit.MultiFitterModel):
    '''
    SU(2) hbxpt extrapolation multifitter class for the proton.
    Includes extrapolation function for the derivative nucleon mass in order to 
    extract the nucleon sigma term
    '''
    def __init__(self, datatag, model_info):
        super(Proton, self).__init__(datatag)
        self.model_info = model_info


    # def fitfcn_sigma_pi(self, p): 
    #     xdata = {}
    #     #xdata['lam_chi'] = p['lam_chi']
    #     if self.model_info['fit_phys_units']:
    #         xdata['eps_pi'] = p['m_pi'] / p['lam_chi']
    #     elif self.model_info['fit_fpi_units']:
    #         xdata['eps_pi'] = p['eps_pi']
    #     if self.model_info['order_disc'] is not None:
    #         xdata['eps2_a'] = p['eps2_a']

    #     if self.model_info['order_strange'] is not None:
    #         xdata['d_eps2_s'] = (2 *p['m_k']**2 - p['m_pi']**2) / p['lam_chi']**2 - 0.3513

    #     output = 0
    #     #if self.model_info['sigma']: # fit to obtain nucleon mass derivative for analytic sigma term #
    #     output += self.fitfcn_lo_sigma(p,xdata)
    #     output += self.fitfcn_nlo_sigma(p,xdata)
    #     output += self.fitfcn_n2lo_sigma(p,xdata)
        
    #     return output

    def fitfcn(self, p,data=None,xdata=None): #data=None):
        if data is not None:
            for key in data.keys():
                p[key] = data[key] 
        if xdata is None:
            xdata = {}
        if 'm_pi' not in xdata:
            xdata['m_pi'] = p['m_pi']
        if 'lam_chi' not in xdata:
            xdata['lam_chi'] = p['lam_chi']
        if self.model_info['fit_phys_units']:
            xdata['eps_pi'] = p['m_pi'] / p['lam_chi']
        elif self.model_info['fit_fpi_units']:
            xdata['eps_pi'] = p['eps_pi']

        xdata['eps_delta'] = (p['m_{delta,0}'] - p['m_{proton,0}']) / p['lam_chi']
        if 'eps2_a' not in xdata:
            xdata['eps2_a'] = p['eps2_a']

        if self.model_info['order_strange'] is not None:
            xdata['d_eps2_s'] = (2 *p['m_k']**2 - p['m_pi']**2) / p['lam_chi']**2 - 0.3513

        output = 0
        if self.model_info['sigma']: # fit to obtain nucleon mass derivative for analytic sigma term #
            output += self.fitfcn_lo_sigma(p,xdata)
            output += self.fitfcn_nlo_sigma(p,xdata)
            output += self.fitfcn_n2lo_sigma(p,xdata)
        elif self.model_info['sigma'] is False:
            # output += self.fitfcn_llo(p,xdata)
            output +=  p['m_{proton,0}']
            output += self.fitfcn_lo(p,xdata) 
            output += self.fitfcn_lo_xpt(p,xdata)
            output += self.fitfcn_nlo_xpt(p,xdata) 
            output += self.fitfcn_n2lo(p,xdata)
            output += self.fitfcn_n2lo_xpt(p,xdata)
            output += self.fitfcn_n4lo(p,xdata)
        return output

    def extrapolate_mass(self,observable=None,p=None, xdata=None):
        if observable == 'sigma_pi':
            return self.fitfcn_sigma_pi(p)

        if observable == 'proton' :
            return self.fitfcn(p)
        
        if observable == 'fpi' :
            return fpi_fit.Fpi.fitfcn(self, p)


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
            if self.model_info['order_chiral'] in ['lo', 'nlo', 'n2lo']: #include chiral log
                output += (xdata['eps_pi']**2) * np.log(xdata['eps_pi']**2)

        elif self.model_info['fit_fpi_units']: # lam_chi dependence OFF #
            if self.model_info['order_disc']    in  ['lo', 'nlo', 'n2lo']:
                output += (p['d_{proton,a}'] * xdata['eps2_a'])
            
            if self.model_info['order_light'] in ['lo','nlo','n2lo','n4lo']:
                output+= (xdata['eps_pi']**2 * p['b_{proton,2}'])

            if self.model_info['order_strange'] in ['lo', 'nlo', 'n2lo']:
                output+= (p['d_{proton,s}'] * xdata['d_eps2_s'])

            if self.model_info['order_chiral'] in ['lo', 'nlo', 'n2lo']: #include chiral log
                output += (xdata['eps_pi']**2) * np.log(xdata['eps_pi']**2)

        return output

    def fitfcn_nlo_xpt(self,p,xdata):
        output = 0
        # if self.model_info['order_chiral'] in ['nlo','n2lo']:
        if self.model_info['xpt']:
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
        if self.model_info['xpt'] is False:
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
                output += (
                p['d_{proton,as}']* xdata['eps2_a'] * xdata['d_eps2_s'] +
                p['d_{proton,ls}'] * xdata['d_eps2_s'] * xdata['eps_pi']**2 + p['d_{proton,ss}'] * xdata['d_eps2_s']**2
            )
            if self.model_info['order_disc'] in ['n2lo','n4lo']:
                output += ( 
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
                output += xdata['eps_pi']**6 *p['b_{proton,6}'] 
            if self.model_info['order_disc'] in ['n4lo']:
                output += (
                + p['d_{proton,all}'] * xdata['eps2_a'] * xdata['eps_pi']**4
                + p['d_{proton,aal}'] * xdata['eps2_a']**3)

            # if self.model_info['order_strange'] in ['n4lo']:
            #     output += 

            # if self.model_info['order_chiral'] in ['n4lo']:
                

        return output

# expansion of nucleon mass derivative for term in analytic sigma term #

    def d_de_lam_chi_lam_chi(self, p,xdata):
            output = 0
            if self.model_info['order_light'] in ['lo','n2lo']:
                output += 2 * xdata['eps_pi']* (p['l4_bar']  - np.log(p['eps_pi']**2) -1 )
            if self.model_info['order_light'] in ['n2lo']:
                output += xdata['eps_pi']**4 * (
                    3/2 * np.log(xdata['eps_pi']**2)**2 + np.log(xdata['eps_pi']**2)*(2*p['c1_F'] + 2*p['l4_bar']+3/2)
                    + 2*p['c2_F'] + p['c1_F'] - p['l4_bar']*(p['l4_bar']-1))

            return output 

    def fitfcn_lo_sigma(self, p, xdata):
        output = 0
        if self.model_info['fit_phys_units']:
            if self.model_info['order_disc'] in ['lo', 'nlo', 'n2lo']:
                output += p['m_{proton,0}'] * (p['d_{proton,a}'] * xdata['eps2_a'])
        
            if self.model_info['order_light'] in ['lo', 'nlo', 'n2lo']:
                    output+= p['b_{proton,2}'] *xdata['eps_pi']* (
                            (self.d_de_lam_chi_lam_chi(p,xdata)*xdata['eps_pi']**2)+
                            (2*xdata['lam_chi']*xdata['eps_pi'])
                    )
            if self.model_info['order_strange'] in ['lo', 'nlo', 'n2lo']:
                output+= p['m_{proton,0}']*(p['d_{proton,s}'] *  xdata['d_eps2_s'])
            
        elif self.model_info['fit_fpi_units']:
            if self.model_info['order_disc'] in ['lo', 'nlo', 'n2lo']:
                output += p['d_{proton,a}'] * xdata['eps2_a']
        
            if self.model_info['order_light'] in ['lo', 'nlo', 'n2lo']:
                    output+= p['b_{proton,2}'] *xdata['eps_pi']
                           
            if self.model_info['order_strange'] in ['lo', 'nlo', 'n2lo']:
                output+= p['d_{proton,s}'] *  xdata['d_eps2_s']

        return output

    def fitfcn_nlo_sigma(self,p,xdata):
        output = 0
        if self.model_info['xpt']:
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
    '''
    expansion of the nucleon mass derivative in order to compute the analytical expression for 
    the sigma_pi_N term. 
    eps_pi(\partial M_N/ \partial eps_pi)
    The term d eps_pi * lam_chi that arises in expansion ->0 since both scalars
    '''
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