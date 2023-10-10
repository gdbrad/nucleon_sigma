import lsqfit
import numpy as np
import gvar as gv
import copy
import functools
# local modules 
import xpt.non_analytic_functions as naf
import xpt.i_o
import xpt.fpi_fit

class FitRoutine():
    '''
    Base lsqfit fitting class for the nucleon and delta mass extrapolations usiing SU(2) hbxpt
    '''

    def __init__(self, prior, data, model_info,phys_point_data):
         
        self.prior = prior
        self.data = data
        self._phys_point_data = phys_point_data
        self.model_info = model_info.copy()
        self._extrapolate = None
        self._simultaneous = False
        self._posterior = None
        y_particles = ['proton','delta']
        data_subset = {part : self.data['m_'+part] for part in y_particles}
        self.y = gv.gvar(dict(gv.mean(data_subset)),dict(gv.evalcov(data_subset)))
        self.models, self.models_dict = self._make_models()
        # self.y = {datatag : self.data['eps_'+datatag] for datatag in self.model_info['particles']}
    
    def __str__(self):
        return str(self.fit)
    
    @functools.cached_property
    def fit(self):
        prior = self._make_prior()
        data = self.y
        fitter = lsqfit.MultiFitter(models=self.models)
        fit = fitter.lsqfit(data=data, prior=prior, fast=False, mopt=False)

        return fit 
    
    def extrapolation(self,observables, p=None, data=None, xdata=None):
        '''chiral extrapolations of baryon mass data using the Feynman-Hellmann theorem to quantify pion mass and strange quark mass dependence of baryon masses. Extrapolations are to the physical point using PDG data. 
        
        Returns(takes a given subset of observables as a list):
        - extrapolated mass (meV)
        - pion sigma term 
        - barred pion sigma term / M_B 
        - strange sigma term
        - barred strange sigma term / M_B 
        '''
        if p is None:
            p = self.posterior
        if data is not None:
            for key in data.keys():
                p[key] = data[key]
        if xdata is None:
            xdata = {}
        if self.model_info['units'] == 'phys':
            xdata['eps_pi'] = p['m_pi'] / p['lam_chi']
        elif self.model_info['units'] == 'fpi':
            xdata['eps_pi'] = p['eps_pi']
        p['l3_bar'] = -1/4 * (
            gv.gvar('3.53(26)') + np.log(xdata['eps_pi']**2))
        p['l4_bar'] =  gv.gvar('4.73(10)')
        p['c2_F'] = gv.gvar(0,20)
        p['c1_F'] = gv.gvar(0,20)
        
        MULTIFIT_DICT = {
            'proton': Proton,
            'delta': Delta
        }
        results = {}

        for particle in self.model_info['particles']:
            model_class = MULTIFIT_DICT.get(particle)
            if model_class is not None:
                model_instance = model_class(datatag=particle, model_info=self.model_info)
        
                results[particle] = {}
                for obs in observables:
                # results = {}

                    output = 0
                # compute the baryon sigma term
                    if obs == 'sigma_pi':
                        if self.model_info['units'] == 'phys':
                            output += model_instance.fitfcn_mass_deriv(p=p,data=data,xdata=xdata)
                            output += xdata['eps_pi']*1/2 *(
                                1 + xdata['eps_pi']**2 *(
                                5/2 - 1/2*p['l3_bar'] - 2*p['l4_bar'])
                                ) * model_instance.fitfcn_mass_deriv(p=p,data=data,xdata=xdata)
                        else:
                            output += model_instance.fitfcn_mass_deriv(p=p,data=data,xdata=xdata)
                    elif obs == 'sigma_bar':
                        if self.model_info['units'] == 'phys':
                            output += xdata['eps_pi']*1/2 *(
                                1 + xdata['eps_pi']**2 *(
                                5/2 - 1/2*p['l3_bar'] - 2*p['l4_bar'])
                                ) * model_instance.fitfcn_mass_deriv(p=p,data=data,xdata=xdata)
                            output = output /  model_instance.fitfcn(p=p)
                        else:
                            output += model_instance.fitfcn_mass_deriv(p=p,data=data,xdata=xdata)
                            output = output / model_instance.fitfcn(p=p)

                    # extrapolate baryon mass to the phys. pt. 
                    elif obs == 'mass':
                        output+= model_instance.fitfcn(p=p)
                    results[particle][obs] = output
        return results
    
    
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

    def _make_models(self, model_info=None):
        if model_info is None:
            model_info = self.model_info.copy()

        model_array = np.array([])
        model_dict = {}

        if 'proton' in model_info['particles']:
            p_model = Proton(datatag='proton', model_info=model_info)
            model_array = np.append(model_array,p_model)
            model_dict['proton'] = p_model
        if 'delta' in model_info['particles']:
            d_model = Delta(datatag='delta', model_info=model_info)
            model_array = np.append(model_array,d_model)
            model_dict['delta'] = d_model

        return model_array,model_dict

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
            for particle in ['proton']:
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
            
class BaseMultiFitterModel(lsqfit.MultiFitterModel):
    """base class for all derived hyperon multifitter classes.
    provides the common `prep_data` routine"""
    def __init__(self, datatag, model_info):
        super(BaseMultiFitterModel,self).__init__(datatag)
        self.model_info = model_info

    def prep_data(self,p,data=None,xdata=None):
        if xdata is None:
            xdata = {}
        if 'm_pi' not in xdata:
            xdata['m_pi'] = p['m_pi']
        if 'lam_chi' not in xdata:
            xdata['lam_chi'] = p['lam_chi']
        if self.model_info['units'] == 'phys':
            xdata['eps_pi'] = p['m_pi'] / p['lam_chi']
        elif self.model_info['units'] == 'fpi':
            xdata['eps_pi'] = p['eps_pi']
        if self.datatag in ['proton']:
            xdata['eps_delta'] = (p['m_{delta,0}'] - p['m_{proton,0}']) / p['lam_chi']
        if 'eps2_a' not in xdata:
            xdata['eps2_a'] = p['eps2_a']
        
        #strange quark mass mistuning
        if self.model_info['order_strange'] is not None:
            xdata['d_eps2_s'] = ((2 * p['m_k']**2 - p['m_pi']**2) / p['lam_chi']**2) - 0.3513

        return xdata
    
    def d_de_lam_chi_lam_chi(self, p,xdata):
        '''
        see eq. 3.32 of Andre's notes. This is the derivative:
        .. math::
            \Lambda_{\Chi} \frac{\partial}{\partial \epsilon_{\pi}} \frac{M_B}{\Lambda_{\Chi}}
        '''
        output = 0
        if self.model_info['order_light'] in ['lo','n2lo']:
            output += 2 * xdata['eps_pi']* (p['l4_bar']  - np.log(p['eps_pi']**2) -1 )
        if self.model_info['order_light'] in ['n2lo']:
            output += xdata['eps_pi']**4 * (
                3/2 * np.log(xdata['eps_pi']**2)**2 + np.log(xdata['eps_pi']**2)*(2*p['c1_F'] + 2*p['l4_bar']+3/2)
                + 2*p['c2_F'] + p['c1_F'] - p['l4_bar']*(p['l4_bar']-1))

        return output 


class Proton(BaseMultiFitterModel):
    '''
    SU(2) hbxpt extrapolation multifitter class for the proton.
    Includes extrapolation function for the derivative nucleon mass in order to 
    extract the nucleon sigma term
    '''
    def __init__(self, datatag, model_info):
        super().__init__(datatag,model_info)
        self.model_info = model_info

    def fitfcn(self, p,data=None,xdata=None): #data=None):
        if data is not None:
            for key in data.keys():
                p[key] = data[key] 
        xdata = self.prep_data(p,data,xdata)
        output = p['m_{proton,0}']
        output += self.fitfcn_lo_ct(p,xdata) 
        output += self.fitfcn_nlo_xpt(p,xdata) 
        output += self.fitfcn_n2lo_ct(p,xdata)
        output += self.fitfcn_n2lo_xpt(p,xdata)
        output += self.fitfcn_n4lo_ct(p,xdata)
        return output
    
    def fitfcn_mass_deriv(self, p, data=None,xdata = None):
        '''
        expansion of the nucleon mass derivative in order to compute the analytical expression for 
        the sigma_pi_N term. 
        eps_pi(\partial M_N/ \partial eps_pi)
        The term d eps_pi * lam_chi that arises in expansion ->0 since both scalars
        '''
        xdata = self.prep_data(p, data, xdata)
        print(xdata)
        
        output = 0 #llo
        output += self.fitfcn_lo_deriv(p,xdata)  
        output += self.fitfcn_nlo_xpt_deriv(p,xdata) 
        output += self.fitfcn_n2lo_ct_deriv(p,xdata)
        output += self.fitfcn_n2lo_xpt_deriv(p,xdata)
        if self.model_info['units'] == 'fpi':
            output *= xdata['lam_chi']
        else:
            return output

    def fitfcn_lo_ct(self, p, xdata):
        ''''pure taylor extrapolation to O(m_pi^2)'''
        output = 0
        # if self.model_info.iincludes(lec='light',order='lo')
        if self.model_info['units'] == 'phys': # lam_chi dependence ON #
            if self.model_info['order_disc'] in  ['lo', 'nlo', 'n2lo']:
                output += p['m_{proton,0}'] * (p['d_{proton,a}'] * xdata['eps2_a'])
        
            if self.model_info['order_light'] in ['lo','nlo','n2lo']:
                output+= p['b_{proton,2}'] * xdata['lam_chi'] * xdata['eps_pi']**2 

            if self.model_info['order_strange'] in ['lo', 'nlo', 'n2lo']:
                output+= p['m_{proton,0}']*   (p['d_{proton,s}'] * xdata['d_eps2_s'])

            if self.model_info['order_chiral'] in ['lo', 'nlo', 'n2lo']: #include chiral log
                output += (xdata['eps_pi']**2) * np.log(xdata['eps_pi']**2)

        elif self.model_info['units'] == 'fpi': # lam_chi dependence OFF #
            if self.model_info['order_disc']    in  ['lo', 'nlo', 'n2lo']:
                output += (p['d_{proton,a}'] * xdata['eps2_a'])
            
            if self.model_info['order_light'] in ['lo','nlo','n2lo','n4lo']:
                output+= (xdata['eps_pi']**2 * p['b_{proton,2}'])

            if self.model_info['order_strange'] in ['lo', 'nlo', 'n2lo']:
                output+= (p['d_{proton,s}'] * xdata['d_eps2_s'])

            if self.model_info['order_chiral'] in ['lo', 'nlo', 'n2lo']: #include chiral log
                output += (p['l4_bar'] + p['b_{proton,2}'])  * (xdata['eps_pi']**2 * p['m_{proton,0}'] * np.log(xdata['eps_pi']**2))


        return output
    
    def fitfcn_lo_deriv(self, p, xdata):
        output = 0
        if self.model_info['units'] == 'phys':
            if self.model_info['order_disc'] in ['lo', 'nlo', 'n2lo']:
                output += p['m_{proton,0}'] * (p['d_{proton,a}'] * xdata['eps2_a'])
        
            if self.model_info['order_light'] in ['lo', 'nlo', 'n2lo']:
                    output+= p['b_{proton,2}'] *xdata['eps_pi']* (
                            (self.d_de_lam_chi_lam_chi(p,xdata)*xdata['eps_pi']**2)+
                            (2*xdata['lam_chi']*xdata['eps_pi'])
                    )
            if self.model_info['order_strange'] in ['lo', 'nlo', 'n2lo']:
                output+= p['m_{proton,0}']*(p['d_{proton,s}'] *  xdata['d_eps2_s'])
            
        elif self.model_info['units'] == 'fpi':
            if self.model_info['order_disc'] in ['lo', 'nlo', 'n2lo']:
                output += p['d_{proton,a}'] * xdata['eps2_a']
        
            if self.model_info['order_light'] in ['lo', 'nlo', 'n2lo']:
                    output+= p['b_{proton,2}'] *xdata['eps_pi']**2
                           
            if self.model_info['order_strange'] in ['lo', 'nlo', 'n2lo']:
                output+= p['d_{proton,s}'] *  xdata['d_eps2_s']

        return output
    
    def fitfcn_nlo_xpt(self,p,xdata):
        """XPT extrapolation to O(m_pi^3)"""

        def compute_phys_output(delta=None):
            """Computes the output for 'phys' units, considering lam_chi dependence."""
            term1 = -3*np.pi/2 * p['g_{proton,proton}']**2 * xdata['lam_chi'] * xdata['eps_pi']**3 
            term2 = 4/3 * p['g_{proton,delta}']**2 * xdata['lam_chi'] * naf.fcn_F(xdata['eps_pi'],xdata['eps_delta'])
            # if delta:
            return term1 - term2
            # return term1

        def compute_fpi_output(delta=None):
            """Computes the output for 'fpi' units, not considering lam_chi dependence."""
            term1 = -3*np.pi/2 * p['g_{proton,proton}']**2 * xdata['eps_pi']**3 
            term2 = 4/3 * p['g_{proton,delta}']**2 * naf.fcn_F(xdata['eps_pi'],xdata['eps_delta'])
            return term1 - term2

        if self.model_info['xpt']:
            if self.model_info['units'] == 'phys':
                if self.model_info['delta']:
                    output = compute_phys_output(delta=True)
                output = compute_phys_output()
            elif self.model_info['units'] == 'fpi':
                if self.model_info['delta']:
                    output = compute_fpi_output(delta=True)
                output = compute_fpi_output()
        else:
            return 0

        return output
        

    def fitfcn_nlo_xpt_deriv(self, p, xdata):
        """Derivative expansion XPT expression at O(m_pi^3)"""

        if not self.model_info['xpt']:
            return 0
        
        def compute_phys_terms():
            term1 = -3/2 * np.pi * p['g_{proton,proton}']**2 * xdata['eps_pi'] * (
                (self.d_de_lam_chi_lam_chi(p, xdata) * xdata['lam_chi']) * xdata['eps_pi']**3 +
                (3 * xdata['lam_chi'] * xdata['eps_pi']**2)
            )
            term2 = -4/3 *p['g_{proton,delta}']**2 * xdata['eps_pi'] * (
                (xdata['lam_chi'] * self.d_de_lam_chi_lam_chi(p, xdata)) * naf.fcn_F(xdata['eps_pi'], xdata['eps_delta']) +
                xdata['lam_chi'] * naf.fcn_dF(xdata['eps_pi'], xdata['eps_delta'])
            )
            return term1 + term2

        def compute_fpi_terms():
            term1 = -3*np.pi/2 * p['g_{proton,proton}']**2 * xdata['eps_pi'] * (3*xdata['lam_chi'] *xdata['eps_pi']**2) 
            term2 = -4/3 * p['g_{proton,delta}']**2 * xdata['eps_pi'] * (xdata['lam_chi']*naf.fcn_dF(xdata['eps_pi'],xdata['eps_delta']))

            return term1+term2
        
        output = 0
        if self.model_info['xpt']:
            if self.model_info['units'] == 'phys':
                output += compute_phys_terms()
            elif self.model_info['units'] == 'fpi':
                output += compute_fpi_terms()

        return output
    
    def fitfcn_n2lo_ct(self,p,xdata):
        """Taylor extrapolation to O(m_pi^4) without terms coming from xpt expressions"""

        def compute_order_strange():
            term1 = p['d_{proton,as}'] * xdata['eps2_a'] * xdata['d_eps2_s']
            term2 = p['d_{proton,ls}'] * xdata['d_eps2_s'] * xdata['eps_pi']**2
            term3 = p['d_{proton,ss}'] * xdata['d_eps2_s']**2

            return term1 + term2 + term3

        def compute_order_disc():
            term1 = p['d_{proton,al}'] * xdata['eps2_a'] * xdata['eps_pi']**2
            term2 = p['d_{proton,aa}'] * xdata['eps2_a']**2

            return term1 + term2

        def compute_order_light():
            return xdata['eps_pi']**4 * p['b_{proton,4}']

        def compute_order_chiral():
            return xdata['eps_pi']**4 * np.log(xdata['eps_pi']**2) * p['a_{proton,4}']

        output = 0

        if self.model_info['units'] == 'phys':  # lam_chi dependence ON 
            if self.model_info['order_strange'] in ['n2lo']:
                output += p['m_{proton,0}'] * compute_order_strange()

            if self.model_info['order_disc'] in ['n2lo']:
                output += p['m_{proton,0}'] * compute_order_disc()

            if self.model_info['order_light'] in ['n2lo']:
                output += xdata['eps_pi']**4 * p['b_{proton,4}'] * xdata['lam_chi']

            if self.model_info['order_chiral'] in ['n2lo']:
                output += xdata['lam_chi'] * compute_order_chiral()

        elif self.model_info['units'] == 'fpi':  # lam_chi dependence ON 
            if self.model_info['order_strange'] in ['n2lo']:
                output += compute_order_strange()

            if self.model_info['order_disc'] in ['n2lo']:
                output += compute_order_disc()

            if self.model_info['order_light'] in ['n2lo']:
                output += compute_order_light()

            if self.model_info['order_chiral'] in ['n2lo']:
                output += compute_order_chiral()

        return output
    
    def fitfcn_n2lo_ct_deriv(self, p, xdata):
        ''''derivative expansion to O(m_pi^4) without terms coming from xpt expressions'''
        def compute_order_strange():
            term1 = p['d_{proton,as}'] * xdata['eps2_a'] * xdata['d_eps2_s']
            term2 = p['d_{proton,ls}'] * xdata['d_eps2_s'] * xdata['eps_pi']**2
            term3 = p['d_{proton,ss}'] * xdata['d_eps2_s']**2

            return term1 + term2 + term3

        def compute_order_disc():
            term1 = p['d_{proton,al}'] * xdata['eps2_a'] * xdata['eps_pi']**2
            term2 = p['d_{proton,aa}'] * xdata['eps2_a']**2

            return term1 + term2

        def compute_order_light(fpi=None): 
            term1 =  p['b_{proton,4}']* xdata['eps_pi']
            term2 =  (self.d_de_lam_chi_lam_chi(p,xdata)*xdata['lam_chi'])*xdata['eps_pi']**4 
            term3 =  4 * xdata['lam_chi'] * xdata['eps_pi']**3

            termfpi = p['a_{proton,4}']* (2* xdata['eps_pi']**4)
            termfpi2 = 2 * p['b_{proton,4}']* xdata['eps_pi']**4
            termfpi3 = p['b_{proton,4}']*(1/4*xdata['eps_pi']**4 - 1/4* p['l3_bar']* xdata['eps_pi']**4)
            if fpi:
                return termfpi + termfpi2 + termfpi3
            else:
                return term1*(term2+term3)

        def compute_order_chiral(fpi=None):
            term1 =  p['a_{proton,4}']* xdata['eps_pi']
            term2 =  (self.d_de_lam_chi_lam_chi(p,xdata)*xdata['lam_chi'])*xdata['eps_pi']**4 * np.log(xdata['eps_pi']**2) 
            term3 = 4 * xdata['lam_chi'] * xdata['eps_pi']**3 * np.log(xdata['eps_pi']**2)
            term4 = 2 * xdata['lam_chi'] * xdata['eps_pi']**3 

            if fpi:
                termfpi = p['a_{proton,4}']* (2*xdata['eps_pi']**4*np.log(xdata['eps_pi']**2))
                termfpi2 = p['g_{proton,4}']*(xdata['eps_pi']**2* naf.fcn_J(xdata['eps_pi'], xdata['eps_delta']) + 1/2* xdata['eps_pi']**3 * naf.fcn_dJ(xdata['eps_pi'], xdata['eps_delta']))
                return termfpi + termfpi2
            else:
                return term1*(term2+term3+term4)
        output = 0

        if self.model_info['units'] == 'phys':  # lam_chi dependence ON 
            if self.model_info['order_strange'] in ['n2lo']:
                output += p['m_{proton,0}'] * compute_order_strange()

            if self.model_info['order_disc'] in ['n2lo']:
                output += p['m_{proton,0}'] * compute_order_disc()

            if self.model_info['order_light'] in ['n2lo']:
                output += compute_order_light()

            if self.model_info['order_chiral'] in ['n2lo']:
                output += compute_order_chiral()

        elif self.model_info['units'] == 'fpi':  # lam_chi dependence ON 
            if self.model_info['order_strange'] in ['n2lo']:
                output += compute_order_strange()

            if self.model_info['order_disc'] in ['n2lo']:
                output += compute_order_disc()

            if self.model_info['order_light'] in ['n2lo']:
                output += compute_order_light(fpi=True)

            if self.model_info['order_chiral'] in ['n2lo']:
                output += compute_order_chiral(fpi=True)

        return output
    

    def fitfcn_n2lo_xpt(self,p,xdata):
        """XPT extrapolation to O(m_pi^4)"""
        def base_term():
            """Computes the base term which is common for both 'phys' and 'fpi' units."""
            return (3/2) * p['g_{proton,4}']** 2 * xdata['eps_pi'] ** 2 * naf.fcn_J(xdata['eps_pi'], xdata['eps_delta'])

        def compute_phys_output():
            """Computes the output for 'phys' units, considering lam_chi dependence."""
            return xdata['lam_chi'] * base_term()

        def compute_fpi_output():
            """Computes the output for 'fpi' units, not considering lam_chi dependence."""
            return base_term()

        if self.model_info['xpt']:
            if self.model_info['units'] == 'phys':
                output = compute_phys_output()
            elif self.model_info['units'] == 'fpi':
                output = compute_fpi_output()
        else:
            return 0

        return output
    
    def fitfcn_n2lo_xpt_deriv(self, p, xdata):
        '''xpt expression for mass derivative expansion at O(m_pi^4)'''

        def base_term():
            """Computes the base term which is common for both 'phys' and 'fpi' units."""
            return p['g_{proton,4}']** 2  * xdata['eps_pi']  

        def compute_phys_output():
            """Computes the output for 'phys' units, considering lam_chi dependence."""
            term1 = (self.d_de_lam_chi_lam_chi(p,xdata)*xdata['lam_chi']) * xdata['eps_pi']** 2 * naf.fcn_J(xdata['eps_pi'], xdata['eps_delta'])
            term2 = 2* xdata['lam_chi'] *xdata['eps_pi']  *  naf.fcn_J(xdata['eps_pi'], xdata['eps_delta'])
            term3 = xdata['lam_chi'] * xdata['eps_pi']**2 *  naf.fcn_dJ(xdata['eps_pi'], xdata['eps_delta'])
            return  base_term() * (term1+term2+term3)

        def compute_fpi_output():
            """Computes the output for 'fpi' units, not considering lam_chi dependence."""
            return base_term()

        if self.model_info['xpt']:
            if self.model_info['units'] == 'phys':
                output = compute_phys_output()
            elif self.model_info['units'] == 'fpi':
                output = compute_fpi_output()
        else:
            return 0

        return output
    


    def fitfcn_n4lo_ct(self,p,xdata):
        """Taylor extrapolation to O(m_pi^6) without terms coming from xpt expressions"""

        def compute_order_strange():
            term1 = p['d_{proton,as}'] * xdata['eps2_a'] * xdata['d_eps2_s']
            term2 = p['d_{proton,ls}'] * xdata['d_eps2_s'] * xdata['eps_pi']**2
            term3 = p['d_{proton,ss}'] * xdata['d_eps2_s']**2

            return term1 + term2 + term3

        def compute_order_disc():
            term1  =  p['d_{proton,all}'] * xdata['eps2_a'] * xdata['eps_pi']**4
            term2  =  p['d_{proton,aal}'] * xdata['eps2_a']**2 * xdata['eps_pi']**2
            term3  =  p['d_{proton,aal}'] * xdata['eps2_a']**3
            return term1 + term2 + term3

        def compute_order_light():
            return xdata['lam_chi'] * (
                xdata['eps_pi']**6 *p['b_{proton,6}'])

        def compute_order_chiral():
            return xdata['eps_pi']**4 * np.log(xdata['eps_pi']**2) * p['a_{proton,4}']

        output = 0

        if self.model_info['units'] == 'phys':  # lam_chi dependence ON 
            if self.model_info['order_strange'] in ['n4lo']:
                output += p['m_{proton,0}'] * compute_order_strange()

            if self.model_info['order_disc'] in ['n4lo']:
                output += p['m_{proton,0}'] * compute_order_disc()

            if self.model_info['order_light'] in ['n4lo']:
                output += xdata['eps_pi']**4 * p['b_{proton,4}'] * xdata['lam_chi']

            if self.model_info['order_chiral'] in ['n4lo']:
                output += xdata['lam_chi'] * compute_order_chiral()

        elif self.model_info['units'] == 'fpi':  # lam_chi dependence ON 
            if self.model_info['order_strange'] in ['n4lo']:
                output += compute_order_strange()

            if self.model_info['order_disc'] in ['n4lo']:
                output += compute_order_disc()

            if self.model_info['order_light'] in ['n4lo']:
                output += compute_order_light()

            if self.model_info['order_chiral'] in ['n4lo']:
                output += compute_order_chiral()

        return output

    def buildprior(self, prior, mopt=False, extend=False):
        return prior

    def builddata(self, data):
        return data[self.datatag]

class Delta(BaseMultiFitterModel):
    def __init__(self, datatag, model_info):
        super(Delta, self).__init__(datatag,model_info)
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

        xdata = self.prep_data(p,data,xdata)
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