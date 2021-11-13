import gvar as gv

prior = {}
# not-even leading order 

prior['m_{proton,0}'] = gv.gvar(0.9,.4)
prior['m_{delta,0}'] = gv.gvar(2,1)

prior['g_{proton,delta}'] = gv.gvar(1.48,5)
prior['g_{proton,proton}'] = gv.gvar(1.27,5)
prior['g_{delta,delta}'] = gv.gvar(-2.2,5)

#FPi FIT#
#prior['l4_bar'] = gv.gvar(4.02,.45)
prior['l4_bar'] = gv.gvar(4.02,4.02)

prior['F0']    = gv.gvar(85,30)
prior['c_2F'] = gv.gvar(0,20)
prior['c_1F'] = gv.gvar(0,20)

prior['d_{fpi,a}']  = gv.gvar(0,4)
prior['d_{fpi,ll}'] = gv.gvar(0,4)
prior['d_{fpi,al}'] = gv.gvar(0,4)
prior['d_{fpi,aa}'] = gv.gvar(0,4)
prior['b_{fpi,2}'] = gv.gvar(0,4)
prior['a_{fpi,2}'] = gv.gvar(0,4)
prior['b_{fpi,4}'] = gv.gvar(0,4)
prior['a_{fpi,4}'] = gv.gvar(0,4)
#prior['c0'] = gv.gvar()

#mpi fit #
prior['M0'] = gv.gvar(139.57,0.18)
prior['B'] = gv.gvar(2900,300)
#prior['l3_bar'] = gv.gvar(3.07,0.64)
prior['d_{mpi,a}']  = gv.gvar(0,4)
prior['d_{mpi,ll}'] = gv.gvar(0,4)
prior['d_{mpi,al}'] = gv.gvar(0,4)
prior['d_{mpi,aa}'] = gv.gvar(0,4)

prior['a_{proton,4}'] = gv.gvar(0, 5)
prior['a_{proton,6}'] = gv.gvar(0, 5)
prior['b_{proton,4}'] = gv.gvar(0,2)
prior['b_{proton,6}'] = gv.gvar(0,2)
prior['b_{proton,2}'] = gv.gvar(2,2)
prior['g_{proton,4}'] = gv.gvar(0,2)
prior['g_{proton,6}'] = gv.gvar(0,5)
prior['d_{proton,a}'] = gv.gvar(0,5)
prior['d_{proton,s}'] = gv.gvar(0,5)
prior['d_{proton,aa}'] = gv.gvar(0,5)
prior['d_{proton,al}'] = gv.gvar(0,5)
prior['d_{proton,as}'] = gv.gvar(0,5)
prior['d_{proton,ls}'] = gv.gvar(0,5)
prior['d_{proton,ss}'] = gv.gvar(0,5)
prior['d_{proton,all}'] = gv.gvar(0,5)
prior['d_{proton,aal}'] =  gv.gvar(0,5)

prior['g_{delta,4}'] = gv.gvar(0,5)
prior['d_{delta,a}'] = gv.gvar(0,5)
prior['b_{delta,4}'] = gv.gvar(0,5)
prior['a_{delta,4}'] = gv.gvar(0,5)
prior['g_{delta,4}'] = gv.gvar(0,5)
prior['d_{delta,aa}'] = gv.gvar(0,5)
prior['d_{delta,al}'] = gv.gvar(0,5)
prior['d_{delta,as}'] = gv.gvar(0,5)
prior['d_{delta,ls}'] = gv.gvar(0,5)
prior['d_{delta,ss}'] = gv.gvar(0,5)
prior['d_{delta,s}'] = gv.gvar(0,5)



