import gvar as gv
import numpy as np
import yaml
import os 
import h5py


import fit_routine as fit
import i_o


class VerboseFitfcn(object):


    def __init__(self, data=None, p=None, model=None, force_gvars=False, phys_point_data=None):
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

        if data is None:
            with open(project_path+'/data/test_data.yaml') as file:
                yaml_file = yaml.safe_load(file)
                data = yaml_file['data']
                if force_gvars: #convert all into gvar 
                    for key in data:
                        data[key] = gv.gvar(data[key], data[key] *1e-5)
        
        if p is None:
            with open(project_path+'/data/test_data.yaml') as file:
                yaml_file = yaml.safe_load(file)
                prior = yaml_file['prior']
                if force_gvars:
                    for key in prior:
                        prior[key] = gv.gvar(prior[key], prior[key] *1e-5)

        if model is None:
            model_description = {}
            model_description['particles'] = ['lambda', 'sigma', 'sigma_st', 'xi', 'xi_st']
            model_description['order_chiral'] = 'n2lo'
            model_description['order_disc'] = 'n2lo'
            model_description['order_strange'] = 'n2lo'
            model_description['xpt'] = True # False => Taylor expansion
            model_description['empirical_bayes'] = False
            model = model_description
        self.phys_point_data = phys_point_data
        self.data = data
        self.prior = prior
        self.model = model
        #self._fitter = xpt.fitter.Fitter(prior=None, data=None, model=None)


    def __str__(self):
        output = ''
        for particle in self.model['particles']:
            #output += '-- ' + particle + '\n'
            #output += self.fitfcn_table(particle)
            output += '\n'
            output += str(self.error_budget(particle))
        return output


    #def _get_prior_keys(self, **kwargs):
    #    return self._fitter.get_prior_keys(**kwargs)

    def error_budget(self,particle):
        output = None
        if particle not in self.model['particles']:
            return None

        #use above dict to fill in values where particle name goes.. leave hardcoded for now
        strange_keys = [
        'd_{lambda,s}','d_{sigma,s}', 'd_{sigma_st,s}', 'd_{xi,s}', 'd_{xi_st,s}',
        'd_{lambda,as}', 'd_{lambda,ls}', 'd_{lambda,ss}', 'd_{sigma,as}', 'd_{sigma,ls}', 'd_{sigma,ss}',
        'd_{sigma_st,as}', 'd_{sigma_st,ls}', 'd_{sigma_st,ss}', 'd_{xi,as}', 'd_{xi,ls}', 'd_{xi,ss}',
        'd_{xi_st,as}', 'd_{xi_st,ls}', 'd_{xi_st,ss}']
        
        chiral_keys = [
        's_{lambda}', 's_{sigma}', 's_{sigma,bar}', 's_{xi}', 's_{xi,bar}', 
        'g_{lambda,sigma}', 'g_{lambda,sigma_st}', 'g_{sigma,sigma}', 'g_{sigma_st,sigma}', 
        'g_{sigma_st,sigma_st}', 'g_{xi,xi}', 'g_{xi_st,xi}', 'g_{xi_st,xi_st}', 'b_{lambda,4}', 
        'b_{sigma,4}', 'b_{sigma_st,4}', 'b_{xi,4}', 'b_{xi_st,4}', 'a_{lambda,4}', 'a_{sigma,4}', 
        'a_{sigma_st,4}', 'a_{xi,4}', 'a_{xi_st,4}'] 
        
        disc_keys = [
        'm_{lambda,0}', 'm_{sigma,0}', 'm_{sigma_st,0}', 'm_{xi,0}', 'm_{xi_st,0}', 'd_{lambda,a}', 'd_{sigma,a}',  
        'd_{sigma_st,a}', 'd_{xi,a}',  'd_{xi_st,a}', 'd_{lambda,aa}', 'd_{lambda,al}', 
        'd_{sigma,aa}', 'd_{sigma,al}',  'd_{sigma_st,aa}', 'd_{sigma_st,al}', 
        'd_{xi,aa}', 'd_{xi,al}',  'd_{xi_st,aa}', 'd_{xi_st,al}']
        
        
        phys_keys = list(self.phys_point_data)
        #stat_keys = ['lam_chi','eps2_a','m_lambda','m_pi','m_k']

        #output = {}
        mdls = fit.fit_routine(prior=self.prior, data=self.data, model_info=self.model)
        #print(mdls.fit)
    
        result = {}
        result['disc'] = mdls.fit.p.partialsdev(
            [self.prior[key] for key in disc_keys if key in self.prior]
        )
        result['chiral'] = mdls.fit.p.partialsdev(
            [self.prior[key] for key in chiral_keys if key in self.prior]
        )
        result['pp_input'] = mdls.fit.p.partialsdev(
            [self.phys_point_data[key] for key in phys_keys]
        )
        # output['stat'] = value.partialsdev(
        #     [self._get_prior(stat_keys), self.fitter.y]
        #     #self.fitter['w0'].y
        # )
        return result


    def _fmt_table(self, values):
        max_len = np.max([len(key) for key in values])

        output = ''
        for key in values:
            output += key.ljust(max_len+2)
            output += str(values[key])
            output += '\n'

        return output  


    def fitfcn_table(self, particle):
        if particle not in self.model['particles']:
            return None

        xdata = {}
        xdata['eps_pi'] = self.data['m_pi'] / self.data['lam_chi']
        xdata['eps2_a'] = self.data['eps2_a']
        xdata['lam_chi'] = self.data['lam_chi']
        if 'm_k' in self.data:
            xdata['d_eps2_s'] = (2 *self.data['m_k']**2 - self.data['m_pi']**2) / self.data['lam_chi']**2 - 0.3513 # pp value

        values = {}
        if particle == 'lambda':
            lsqfit_model = fit.Lambda(datatag=None, model_info=self.model)

            if 'm_{sigma,0}' in self.p:
                xdata['eps_sigma'] = (self.p['m_{sigma,0}'] - self.p['m_{lambda,0}']) / self.data['lam_chi']
            if 'm_{sigma_st,0}' in self.p:
                xdata['eps_sigma_st'] = (self.p['m_{sigma_st,0}'] - self.p['m_{lambda,0}']) / self.data['lam_chi']

            values['LLO:ct'] = self.p['m_{lambda,0}']

        elif particle == 'sigma':
            lsqfit_model = fit.Sigma(datatag=None, model_info=self.model)

            if 'm_{lambda,0}' in self.p:
                xdata['eps_lambda'] = (self.p['m_{sigma,0}'] - self.p['m_{lambda,0}']) / self.data['lam_chi']
            if 'm_{sigma_st,0}' in self.p:
                xdata['eps_sigma_st'] = (self.p['m_{sigma_st,0}'] - self.p['m_{sigma,0}']) / self.data['lam_chi']

            values['LLO:ct'] = self.p['m_{sigma,0}']

        elif particle == 'sigma_st':
            lsqfit_model = fit.Sigma_st(datatag=None, model_info=self.model)

            if 'm_{lambda,0}' in self.p:
                xdata['eps_lambda'] = (self.p['m_{sigma_st,0}'] - self.p['m_{lambda,0}']) / self.data['lam_chi']
            if 'm_{sigma,0}' in self.p:
                xdata['eps_sigma'] = (self.p['m_{sigma_st,0}'] - self.p['m_{sigma,0}']) / self.data['lam_chi']

            values['LLO:ct'] = self.p['m_{sigma_st,0}']

        elif particle == 'xi':
            lsqfit_model = fit.Xi(datatag=None, model_info=self.model)
            if 'm_{xi_st,0}' in self.p:
                xdata['eps_delta'] = (self.p['m_{xi_st,0}'] - self.p['m_{xi,0}']) / self.data['lam_chi']

            values['LLO:ct'] = self.p['m_{xi,0}']

        elif particle == 'xi_st':
            lsqfit_model = fit.Xi_st(datatag=None, model_info=self.model)

            if 'm_{xi,0}' in self.p:
                xdata['eps_delta'] = (self.p['m_{xi_st,0}'] - self.p['m_{xi,0}']) / self.data['lam_chi']

            values['LLO:ct'] = self.p['m_{xi_st,0}']

        values['LO:ct'] = lsqfit_model.fitfcn_lo_ct(p=self.p, xdata=xdata)
        values['NLO:xpt'] = lsqfit_model.fitfcn_nlo_xpt(p=self.p, xdata=xdata)
        values['N2LO:ct'] = lsqfit_model.fitfcn_n2lo_ct(p=self.p, xdata=xdata)
        values['N2LO:xpt'] = lsqfit_model.fitfcn_n2lo_xpt(p=self.p, xdata=xdata)
        values['total'] = np.sum([values[k] for k in values])

        return self._fmt_table(values)
