import gvar as gv

prior = {}
# not-even leading order 

prior['m_{proton,0}'] = gv.gvar(1,1)
prior['m_{delta,0}'] = gv.gvar(2,1)

prior['g_{proton,delta}'] = gv.gvar(0.91,5)
prior['g_{proton,proton}'] = gv.gvar(1.27,5)
prior['g_{delta,delta}'] = gv.gvar(0.59,5)

# n2lo

prior['a_{proton,4}'] = gv.gvar(0, 5)
prior['a_{proton,6}'] = gv.gvar(0, 5)
prior['b_{proton,4}'] = gv.gvar(0,5)
prior['b_{proton,6}'] = gv.gvar(0,5)
prior['b_{proton,2}'] = gv.gvar(0, 5)
prior['g_{proton,4}'] = gv.gvar(0,5)
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



