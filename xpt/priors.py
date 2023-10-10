import gvar as gv
import numpy as np

def get_prior(units=None):
    if units == 'phys':
        gs_baryons = {
            'm_{proton,0}':gv.gvar(1000,2),
            'm_{delta,0}': gv.gvar(1232, 2)
        }
    elif units =='lam_chi':
        gs_baryons = {
            'm_{proton,0}': gv.gvar(1,1),
            'm_{delta,0}': gv.gvar(2,1),
        }
    else:
        raise ValueError(f"Invalid units: {units}")
    
    prior = {
        **gs_baryons,
        'g_{proton,delta}' : gv.gvar(1.48,5),
        'g_{proton,proton}' : gv.gvar(1.27,5),
        'g_{delta,delta}' : gv.gvar(-2.2,5),

        #FPi FIT#
        #'l4_bar' : gv.gvar(4.02,.45),
        'l4_bar' : gv.gvar(4.02,4.02),

        'F0'    : gv.gvar(85,30),
        'c2_F' : gv.gvar(0,20),
        'c1_F' : gv.gvar(0,20),

        'd_{fpi,a}'  : gv.gvar(0,4),
        'd_{fpi,ll}' : gv.gvar(0,4),
        'd_{fpi,al}' : gv.gvar(0,4),
        'd_{fpi,aa}' : gv.gvar(0,4),
        'b_{fpi,2}' : gv.gvar(0,4),
        'a_{fpi,2}' : gv.gvar(0,4),
        'b_{fpi,4}' : gv.gvar(0,4),
        'a_{fpi,4}' : gv.gvar(0,4),
        #'c0' : gv.gvar(),

        #mpi fit #
        'M0' : gv.gvar(139.57,0.18),
        'B' : gv.gvar(2900,300),
        #'l3_bar' : gv.gvar(3.07,0.64),
        'd_{mpi,a}'  : gv.gvar(0,4),
        'd_{mpi,ll}' : gv.gvar(0,4),
        'd_{mpi,al}' : gv.gvar(0,4),
        'd_{mpi,aa}' : gv.gvar(0,4),

        'a_{proton,4}' : gv.gvar(0, 5),
        'a_{proton,6}' : gv.gvar(0, 5),
        'b_{proton,4}' : gv.gvar(0,2),
        'b_{proton,6}' : gv.gvar(0,2),
        'b_{proton,2}' : gv.gvar(2,2),
        'g_{proton,4}' : gv.gvar(0,2),
        'g_{proton,6}' : gv.gvar(0,5),
        'd_{proton,a}' : gv.gvar(0,5),
        'd_{proton,s}' : gv.gvar(0,5),
        'd_{proton,aa}' : gv.gvar(0,5),
        'd_{proton,al}' : gv.gvar(0,5),
        'd_{proton,as}' : gv.gvar(0,5),
        'd_{proton,ls}' : gv.gvar(0,5),
        'd_{proton,ss}' : gv.gvar(0,5),
        'd_{proton,all}' : gv.gvar(0,5),
        'd_{proton,aal}' :  gv.gvar(0,5),

        'g_{delta,4}' : gv.gvar(0,5),
        'd_{delta,a}' : gv.gvar(0,5),
        'b_{delta,4}' : gv.gvar(0,5),
        'b_{delta,2}' : gv.gvar(0,5),
        'a_{delta,4}' : gv.gvar(0,5),
        'd_{delta,aa}' : gv.gvar(0,5),
        'd_{delta,al}' : gv.gvar(0,5),
        'd_{delta,as}' : gv.gvar(0,5),
        'd_{delta,ls}' : gv.gvar(0,5),
        'd_{delta,ss}' : gv.gvar(0,5),
        'd_{delta,s}' : gv.gvar(0,5),
    }

    return prior

