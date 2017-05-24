#Data set 1
u1str  = '0.5*t*x[0]*x[0]'
u2str  = '0.5*t*x[1]*x[1]'
z1str  = '-cos(x[0]+x[1])*exp(t)'
z2str  = '-cos(x[0]+x[1])*exp(t)'
pstr   = 'exp(t)*sin(x[0]+x[1])'
f1str  = 'lmbda*t + 2*mu*t - alpha*exp(t)*cos(x[0]+x[1])'
f2str  = 'lmbda*t + 2*mu*t - alpha*exp(t)*cos(x[0]+x[1])'
gstr   = '(c+2)*exp(t)*sin(x[0]+x[1]) + alpha*(x[0]+x[1])'
gmstr  = '0.0*t'
s11str = 'lmbda*(t*x[0]+t*x[1])-alpha*exp(t)*sin(x[0]+x[1])+mu*t*x[0]*2.0'
s12str = '0.0'
s21str = '0.0'
s22str = 'lmbda*(t*x[0]+t*x[1])-alpha*exp(t)*sin(x[0]+x[1])+mu*t*x[1]*2.0'
#Hack for stress bc
ps11str = 'exp(t)*sin(x[0]+x[1])+lmbda*(t*x[0]+t*x[1])-alpha*exp(t)*sin(x[0]+x[1])+mu*t*x[0]*2.0'
ps12str = '0.0'
ps21str = '0.0'
ps22str = 'exp(t)*sin(x[0]+x[1])+lmbda*(t*x[0]+t*x[1])-alpha*exp(t)*sin(x[0]+x[1])+mu*t*x[1]*2.0'

#Data set 2
# u1str = 'x[0]*x[1]*sin(pi*t)*(x[0]-1.0)*(x[1]-1.0)'
# u2str = 'x[0]*x[1]*sin(pi*t)*(x[0]-1.0)*(x[1]-1.0)'

# z1str = 'K11*pi*t*cos(pi*x[0]*2.0)*sin(pi*x[1]*2.0)*-2.0-K12*pi*t*cos(pi*x[1]*2.0)*sin(pi*x[0]*2.0)*2.0'
# z2str = 'K21*pi*t*cos(pi*x[0]*2.0)*sin(pi*x[1]*2.0)*-2.0-K22*pi*t*cos(pi*x[1]*2.0)*sin(pi*x[0]*2.0)*2.0'

# pstr  = 't*sin(pi*x[0]*2.0)*sin(pi*x[1]*2.0)'

# s11str='mu*(x[1]*sin(pi*t)*(x[0]-1.0)*(x[1]-1.0)+x[0]*x[1]*sin(pi*t)*(x[1]-1.0))*2.0+lmbda*(x[0]*sin(pi*t)*(x[0]-1.0)*(x[1]-1.0)+x[1]*sin(pi*t)*(x[0]-1.0)*(x[1]-1.0)+x[0]*x[1]*sin(pi*t)*(x[0]-1.0)+x[0]*x[1]*sin(pi*t)*(x[1]-1.0))-alpha*t*sin(pi*x[0]*2.0)*sin(pi*x[1]*2.0)'
# s12str='mu*(x[0]*sin(pi*t)*(x[0]-1.0)*(x[1]-1.0)*(1.0/2.0)+x[1]*sin(pi*t)*(x[0]-1.0)*(x[1]-1.0)*(1.0/2.0)+x[0]*x[1]*sin(pi*t)*(x[0]-1.0)*(1.0/2.0)+x[0]*x[1]*sin(pi*t)*(x[1]-1.0)*(1.0/2.0))*2.0'
# s21str='mu*(x[0]*sin(pi*t)*(x[0]-1.0)*(x[1]-1.0)*(1.0/2.0)+x[1]*sin(pi*t)*(x[0]-1.0)*(x[1]-1.0)*(1.0/2.0)+x[0]*x[1]*sin(pi*t)*(x[0]-1.0)*(1.0/2.0)+x[0]*x[1]*sin(pi*t)*(x[1]-1.0)*(1.0/2.0))*2.0'
# s22str='mu*(x[0]*sin(pi*t)*(x[0]-1.0)*(x[1]-1.0)+x[0]*x[1]*sin(pi*t)*(x[0]-1.0))*2.0+lmbda*(x[0]*sin(pi*t)*(x[0]-1.0)*(x[1]-1.0)+x[1]*sin(pi*t)*(x[0]-1.0)*(x[1]-1.0)+x[0]*x[1]*sin(pi*t)*(x[0]-1.0)+x[0]*x[1]*sin(pi*t)*(x[1]-1.0))-alpha*t*sin(pi*x[0]*2.0)*sin(pi*x[1]*2.0)'

# ps11str='t*sin(pi*x[0]*2.0)*sin(pi*x[1]*2.0) + mu*(x[1]*sin(pi*t)*(x[0]-1.0)*(x[1]-1.0)+x[0]*x[1]*sin(pi*t)*(x[1]-1.0))*2.0+lmbda*(x[0]*sin(pi*t)*(x[0]-1.0)*(x[1]-1.0)+x[1]*sin(pi*t)*(x[0]-1.0)*(x[1]-1.0)+x[0]*x[1]*sin(pi*t)*(x[0]-1.0)+x[0]*x[1]*sin(pi*t)*(x[1]-1.0))-alpha*t*sin(pi*x[0]*2.0)*sin(pi*x[1]*2.0)'
# ps12str='mu*(x[0]*sin(pi*t)*(x[0]-1.0)*(x[1]-1.0)*(1.0/2.0)+x[1]*sin(pi*t)*(x[0]-1.0)*(x[1]-1.0)*(1.0/2.0)+x[0]*x[1]*sin(pi*t)*(x[0]-1.0)*(1.0/2.0)+x[0]*x[1]*sin(pi*t)*(x[1]-1.0)*(1.0/2.0))*2.0'
# ps21str='mu*(x[0]*sin(pi*t)*(x[0]-1.0)*(x[1]-1.0)*(1.0/2.0)+x[1]*sin(pi*t)*(x[0]-1.0)*(x[1]-1.0)*(1.0/2.0)+x[0]*x[1]*sin(pi*t)*(x[0]-1.0)*(1.0/2.0)+x[0]*x[1]*sin(pi*t)*(x[1]-1.0)*(1.0/2.0))*2.0'
# ps22str='t*sin(pi*x[0]*2.0)*sin(pi*x[1]*2.0) + mu*(x[0]*sin(pi*t)*(x[0]-1.0)*(x[1]-1.0)+x[0]*x[1]*sin(pi*t)*(x[0]-1.0))*2.0+lmbda*(x[0]*sin(pi*t)*(x[0]-1.0)*(x[1]-1.0)+x[1]*sin(pi*t)*(x[0]-1.0)*(x[1]-1.0)+x[0]*x[1]*sin(pi*t)*(x[0]-1.0)+x[0]*x[1]*sin(pi*t)*(x[1]-1.0))-alpha*t*sin(pi*x[0]*2.0)*sin(pi*x[1]*2.0)'

# gmstr = '(sin(pi*t)*(2*x[0]*x[0]*x[1] - x[0]*x[0] - 2*x[0]*x[1]*x[1] + x[0] + x[1]*x[1] - x[1]))/2'

# f1str = 'mu*(x[0]*sin(pi*t)*(x[0]-1.0)+x[0]*sin(pi*t)*(x[1]-1.0)*(1.0/2.0)+x[1]*sin(pi*t)*(x[0]-1.0)*(1.0/2.0)+sin(pi*t)*(x[0]-1.0)*(x[1]-1.0)*(1.0/2.0)+x[0]*x[1]*sin(pi*t)*(1.0/2.0))*2.0+lmbda*(x[0]*sin(pi*t)*(x[1]-1.0)+x[1]*sin(pi*t)*(x[0]-1.0)+x[1]*sin(pi*t)*(x[1]-1.0)*2.0+sin(pi*t)*(x[0]-1.0)*(x[1]-1.0)+x[0]*x[1]*sin(pi*t))+mu*x[1]*sin(pi*t)*(x[1]-1.0)*4.0-pi*alpha*t*cos(pi*x[0]*2.0)*sin(pi*x[1]*2.0)*2.0'
# f2str = 'lmbda*(x[0]*sin(pi*t)*(x[0]-1.0)*2.0+x[0]*sin(pi*t)*(x[1]-1.0)+x[1]*sin(pi*t)*(x[0]-1.0)+sin(pi*t)*(x[0]-1.0)*(x[1]-1.0)+x[0]*x[1]*sin(pi*t))+mu*(x[0]*sin(pi*t)*(x[1]-1.0)*(1.0/2.0)+x[1]*sin(pi*t)*(x[0]-1.0)*(1.0/2.0)+x[1]*sin(pi*t)*(x[1]-1.0)+sin(pi*t)*(x[0]-1.0)*(x[1]-1.0)*(1.0/2.0)+x[0]*x[1]*sin(pi*t)*(1.0/2.0))*2.0+mu*x[0]*sin(pi*t)*(x[0]-1.0)*4.0-pi*alpha*t*cos(pi*x[1]*2.0)*sin(pi*x[0]*2.0)*2.0'

# gstr  = 'alpha*(pi*x[0]*x[1]*cos(pi*t)*(x[1]-1.0)+pi*x[0]*cos(pi*t)*(x[0]-1.0)*(x[1]-1.0)+pi*x[1]*cos(pi*t)*(x[0]-1.0)*(x[1]-1.0)+pi*x[0]*x[1]*cos(pi*t)*(x[0]-1.0))+c*sin(pi*x[0]*2.0)*sin(pi*x[1]*2.0)+K11*(pi*pi)*t*sin(pi*x[0]*2.0)*sin(pi*x[1]*2.0)*4.0+K22*(pi*pi)*t*sin(pi*x[0]*2.0)*sin(pi*x[1]*2.0)*4.0-K12*(pi*pi)*t*cos(pi*x[0]*2.0)*cos(pi*x[1]*2.0)*4.0-K21*(pi*pi)*t*cos(pi*x[0]*2.0)*cos(pi*x[1]*2.0)*4.0'

