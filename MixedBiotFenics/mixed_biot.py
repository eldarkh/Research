from dolfin import *
from dataset import *
from savelatex import *
from mshr import *

prm = parameters["form_compiler"]
prm["representation"] = 'quadrature'

plotflag = True
printflag = False

latexflag = True
ltxname1 = 'set1.txt'
ltxname2 = 'set2.txt'

trapqflag = False

# Boundary conditions on entire boundary (stress,displ,vel,pres)
# Standard sets of BC
bcflag = [0,0,0,0]
if True:
	bcflag = [0,1,0,1]	# Displ  & pres
	#bcflag = [0,1,1,0]	# Displ  & flux
	#bcflag = [1,0,0,1]	# Stress & pres (not implemented)
	#bcflag = [1,0,1,0]  # Stress & flux (not implemented)
	assert( (bcflag[0] + bcflag[1] > 0 ) and (bcflag[2] + bcflag[3]> 0)), 'Unsupported type of BC'

e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11 = ([] for i in range (11))
r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11 = ([] for i in range (11))

[i.append(0) for i in (r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11)]

def solve_problem(nx):
	# Initialize errors
	zL2Hdiv, zrL2Hdiv, pL2L2, pL2L2, prL2L2, sL2Hdiv, srL2Hdiv, \
		uL2L2, urL2L2, gL2L2, grL2L2, psL2, uL2, gL2, pL2, zL2, sL2 = (0 for i in range (17))

	# Geometry
	domain = Rectangle(Point(0.,0.),Point(1.,1.))
	mesh = generate_mesh(domain,nx)

	n = FacetNormal(mesh)

	# Define boundaries separately for use in natural BC
	class Right(SubDomain):
		def inside(self,x,on_boundary):
			return near(x[0],1.0)

	class Left(SubDomain):
		def inside(self,x,on_boundary):
			return near(x[0],0.0)
	class Bottom(SubDomain):
		def inside(self,x,on_boundary):
			return near(x[1],0.0)

	class Top(SubDomain):
		def inside(self,x,on_boundary):
			return near(x[1],1.0)

	# Instantiate boundaries
	right = Right()
	left = Left()
	top = Top()
	bottom = Bottom()

	# Mark boundaries
	boundaries = FacetFunction("size_t", mesh)
	boundaries.set_all(0) 
	bottom.mark(boundaries, 1)
	right.mark(boundaries, 2)
	top.mark(boundaries,3)
	left.mark(boundaries,4)

	# Mark measures
	ds = Measure("ds")[boundaries]

	def bdry(x,on_boundary):
		return on_boundary

	# Physical parameters
	dt = 0.0001
	T = 0.002
	t = 0.0

	K11 = 1.0
	K12 = 0.0
	K21 = 0.0
	K22 = 1.0
	K = Expression( ((str(K11),str(K12)), (str(K21),str(K22))) )

	E = 1.0
	nu = 0.2

	lmbda = E*nu/((1+nu)*(1-2*nu))
	mu = E/(2*(1+nu))

	c0 = Constant(1.0)
	alpha = Constant(1.0)

	# Spaces
	S = VectorFunctionSpace(mesh, 'BDM', 1)
	V = VectorFunctionSpace(mesh, 'DG',0)
	Q = FunctionSpace(mesh, 'CG', 1)
	Z = FunctionSpace(mesh, 'BDM', 1)
	W = FunctionSpace(mesh, 'DG', 0)
	M = MixedFunctionSpace([S,V,Q,Z,W])

	Vplot = FunctionSpace(mesh, 'DG', 0)
	Qplot = FunctionSpace(mesh, 'CG', 1)
	Zplot = FunctionSpace(mesh, 'BDM', 1)
	Wplot = FunctionSpace(mesh, 'DG', 0)

	# Trial and test functions
	(sigma,u,gmma,z,p) = TrialFunctions(M)
	(tau,v,ksi,q,w) = TestFunctions(M)

	# To store current and previous solutions
	sol = Function(M)
	sol_prev = Function(M)

	# Analytical solutions
	u0 = Expression((u1str,u2str), t=0)
	u0prev = Expression((u1str,u2str), t=0)
	z0 = Expression((z1str,z2str), t=0, K11=K11, K12=K12, K21=K21, K22=K22)
	p0 = Expression(pstr, t=0)
	gm0 = Expression(gmstr, t=0)
	sigma0 = Expression(((s11str,s12str),(s21str,s22str)), t=0, mu=mu, lmbda=lmbda, alpha=float(alpha))
	s10 = Expression((s11str, s12str),t=0, mu=mu, lmbda=lmbda, alpha=float(alpha))
	s20 = Expression((s21str, s22str),t=0, mu=mu, lmbda=lmbda, alpha=float(alpha))
	psigma0 = Expression(((ps11str,ps12str),(ps21str,ps22str)), t=0, mu=mu, lmbda=lmbda, alpha=float(alpha))
	ps10 = Expression((ps11str, ps12str),t=0, mu=mu, lmbda=lmbda, alpha=float(alpha))
	ps20 = Expression((ps21str, ps22str),t=0, mu=mu, lmbda=lmbda, alpha=float(alpha))

	# Source and sink terms
	f = Expression((f1str,f2str), t=0, mu=mu, lmbda=lmbda, alpha=float(alpha))
	g = Expression(gstr, t=0, mu=mu, lmbda=lmbda, c=float(c0), alpha=float(alpha), K11=K11, K12=K12, K21=K21, K22=K22)

	# For better plotting
	u0f = interpolate(u0,V)
	z0f = interpolate(z0,Z)
	p0f = interpolate(p0,W)
	s0f = interpolate(sigma0,S)
	gm0f= interpolate(gm0,Q)
	ps0f = interpolate(psigma0,S)

	# Initial conditions
	uprev = u0f
	pprev = p0f
	sprev = s0f

	# Specify quadrature rules (dx2 for default)
	dx2 = dx(None, {'quadrature_rule': 'default', 'quadrature_degree': 3})
	dx3 = dx(None, {'quadrature_rule': 'default', 'quadrature_degree': 1})
	dx1 = dx(None, {'quadrature_rule': 'vertex'})

	# Switch quad rules
	if not trapqflag:
		dx1 = dx2

	# Utilities
	def asym(z):
		return z[0,1] - z[1,0]

	def operatorA(s):
		return 1/(2*mu)*(s - lmbda/(2*mu + 2*lmbda)*tr(s)*Identity(2))

	def pI(s):
	        return s*Identity(2)

	# Variational formulation
	# Darcy terms
	a00  = inner(inv(K)*z,q)*dx1
	a01  = -inner(p, div(q))*dx2
	a02  = inner(div(z),w)*dx2

	# Elasticity
	a70 = inner(operatorA(sigma),tau)*dx1
	a71 = inner(u,div(tau))*dx2
	a72 = inner(gmma, asym(tau))*dx1
	a73 = inner(div(sigma),v)*dx2
	a74 = inner(asym(sigma),ksi)*dx1
	a75 = inner(operatorA(pI(p)),tau)*dx1

	# Biot term in last equation
	a88 =  inner(tr(operatorA(sigma)),w)*dx2 + inner(tr(operatorA(pI(p))),w)*dx2

	# Time dependent
	m00 = float(c0)*inner(p,w)*dx2

	a = dt*(a00 + a01 + a02) + dt*(a70 + a71 + a72 + a73 + a74 + float(alpha)*a75) + m00 + float(alpha)*a88
	A = assemble(a)

	# To plot solutions
	count = 0
	while (t<=T+DOLFIN_EPS):
		t += dt
		u0.t, u0prev.t, z0.t, p0.t, f.t, g.t, gm0.t, sigma0.t, \
			psigma0.t, s10.t, s20.t, ps10.t, ps20.t = (t for i in range(13))

		# RHS
		l1 = inner(u0, tau*n)*ds
		l2 = inner(f,v)*dx2
		l3 = 0.0
		l4 = -inner(p0*n,q)*ds
		l5 = inner(g,w)*dx2
	
		# Due to time discretization
		l7 = float(c0)*inner(pprev,w)*dx2
		l8 = inner(tr(operatorA(sprev)),w)*dx2 + inner(tr(operatorA(pI(pprev))),w)*dx2

		# Assemble RHS
		L = dt*(bcflag[1]*l1 + l2 + l3 + bcflag[3]*l4 + l5) + l7 + l8
		b = assemble(L)

		# Essential boundary conditions for stress (WILL NOT WORK WHEN TWO SIDES SHARE TRIANGLE)
		bc1left   = DirichletBC(M.sub(0), sigma0, boundaries, 4)
		bc1right  = DirichletBC(M.sub(0), sigma0, boundaries, 2)
		bc1top    = DirichletBC(M.sub(0), sigma0, boundaries, 3)
		bc1bottom = DirichletBC(M.sub(0), sigma0, boundaries, 1)
		# Essential boundary conditions for flux
		bc2 	  = DirichletBC(M.sub(3), z0, bdry)
		bc2left   = DirichletBC(M.sub(3), z0, boundaries, 4)
		bc2right  = DirichletBC(M.sub(3), z0, boundaries, 2)
		bc2top    = DirichletBC(M.sub(3), z0, boundaries, 3)
		bc2bottom = DirichletBC(M.sub(3), z0, boundaries, 1)

		if bcflag[0]: # NOT USED ACTUALLY 
			#bcleft.apply(A,b)
			#bcright.apply(A,b)
			#bctop.apply(A,b)
			bcbottom.apply(A,b)
		elif bcflag[2]:
			bc2.apply(A,b)
			#bc2left.apply(A,b)
			#bc2right.apply(A,b)
			#bc2top.apply(A,b)
			#bc2bottom.apply(A,b)

		# For test case with all BC used
		#bc1bottom.apply(A,b)
		#bc2top.apply(A,b)
		#bc2bottom.apply(A,b)

		# Solve
		solve(A, sol.vector(), b)
		(sprev,uprev,gmmaprev,zprev,pprev) = sol.split()
		(s1p, s2p) = sprev.split(deepcopy=True)
	
		# To plot true
		VV = FunctionSpace(mesh,'BDM',1)
		u0f = interpolate(u0,VV)
		z0f = interpolate(z0,Z)
		p0f = interpolate(p0,W)
		s0f = interpolate(sigma0,S)
		psigma0f = interpolate(psigma0,S)
		gm0f = interpolate(gm0,Q)
		s10f = interpolate(s10,Z)
		s20f = interpolate(s20,Z)
		ps10f = interpolate(ps10,Z)
		ps20f = interpolate(ps20,Z)

		psprev = project(sprev + pprev*Identity(2), S)
		(ps1p, ps2p) = psprev.split(deepcopy=True)

		# Plot computed
		if plotflag:
			File('pictures/displ'+str(count)+'.pvd') << uprev
			File('pictures/velocity'+str(count)+'.pvd') << zprev
			File('pictures/pressure'+str(count)+'.pvd') << pprev
			File('pictures/stress1'+str(count)+'.pvd') << s1p
			File('pictures/stress2'+str(count)+'.pvd') << s2p
			File('pictures/Pstress1'+str(count)+'.pvd') << ps1p
			File('pictures/Pstress2'+str(count)+'.pvd') << ps2p
			File('pictures/rotation'+str(count)+'.pvd') << gmmaprev

		(sf,uf,gmf,zf,pf) = sol.split()

		# Errors
		eVelL2    = assemble( inner(zf-z0f,zf-z0f)*dx )
		eVelDiv   = assemble( inner(div(zf-z0f),div(zf-z0f))*dx )
		ePres     = assemble( inner(pf-p0f,pf-p0f)*dx )
		eSigmaL2  = assemble( inner(sf-s0f,sf-s0f)*dx )
		eSigmaDiv = assemble( inner(div(sf-s0f),div(sf-s0f))*dx )
		eDispl    = assemble( inner(uf-u0f,uf-u0f)*dx )
		eRot      = assemble( inner(gmf-gm0f,gmf-gm0f)*dx )
		ePsigmaL2 = assemble( inner(sf + pf*Identity(2) - psigma0f, sf + pf*Identity(2) - psigma0f)*dx )

		# Abs values
		rVelL2    = assemble( inner(z0f,z0f)*dx )
		rVelDiv   = assemble( inner(div(z0f),div(z0f))*dx )
		rPres     = assemble( inner(p0f,p0f)*dx )
		rSigmaL2  = assemble( inner(s0f,s0f)*dx )
		rSigmaDiv = assemble( inner(s0f,s0f)*dx )
		rDispl    = assemble( inner(u0f,u0f)*dx )
		rRot      = 1	# this particular solution has gamma = 0
		rPsigmaL2 = assemble( inner(psigma0f, psigma0f)*dx )

		# L2 in time norms
		zL2Hdiv  += eVelL2 + eVelDiv
		zrL2Hdiv += rVelL2 + rVelDiv
		
		pL2L2    += ePres
		prL2L2   += rPres
		
		sL2Hdiv  += eSigmaDiv + eSigmaL2
		srL2Hdiv += rSigmaDiv + rSigmaL2
		
		uL2L2    += eDispl
		urL2L2   += rDispl
		
		gL2L2    += eRot
		grL2L2   += 0	# this particular solution has gamma = 0

		# Linf in time norms
		if psL2 <= sqrt(ePsigmaL2)/sqrt(rPsigmaL2) and count > 0:
			psL2 = sqrt(ePsigmaL2)/sqrt(rPsigmaL2)

		if sL2 <= sqrt(eSigmaL2)/sqrt(rSigmaL2) and count > 0:
			sL2 = sqrt(eSigmaL2)/sqrt(rSigmaL2)

		if uL2 <= sqrt(eDispl)/sqrt(rDispl) and count > 0:
			uL2 = sqrt(eDispl)/sqrt(rDispl)

		if gL2 <= sqrt(eRot) and count > 0:
			gL2 = sqrt(eRot)

		if pL2 <= sqrt(ePres)/sqrt(rPres) and count > 0:
			pL2 = sqrt(ePres)/sqrt(rPres)		

		if zL2 <= sqrt(eVelL2)/sqrt(rVelL2) and count > 0:
			zL2 = sqrt(eVelL2)/sqrt(rVelL2)

		count += 1
		"""end of time loop"""

	# Final step solutions
	(sf,uf,gmf,zf,pf) = sol.split()

	# Darcy
	e1.append(sqrt(zL2Hdiv)/sqrt(zrL2Hdiv))
	e2.append(zL2)
	e3.append(sqrt(pL2L2)/sqrt(prL2L2))
	e4.append(pL2)

	# Elasticity
	e5.append(sqrt(sL2Hdiv)/sqrt(srL2Hdiv))
	e6.append(sL2)
	e7.append(psL2)
	e8.append(sqrt(uL2L2)/sqrt(urL2L2))
	e9.append(uL2)
	e10.append(sqrt(gL2L2))
	e11.append(gL2)

	print 'Finished computing for N =', nx
	"""end of h loop"""
""" end of function solve_problem """

N = [2,4,8,16,32]
for nx in N:
	solve_problem(nx)

# Compute and output rates
if len(N) > 1:
	for i in range (1,len(e1)):
		r1.append(-ln(e1[i]/e1[i-1])/ln(N[i]/N[i-1]))
		r2.append(-ln(e2[i]/e2[i-1])/ln(N[i]/N[i-1]))
		r3.append(-ln(e3[i]/e3[i-1])/ln(N[i]/N[i-1]))
		r4.append(-ln(e4[i]/e4[i-1])/ln(N[i]/N[i-1]))
		r5.append(-ln(e5[i]/e5[i-1])/ln(N[i]/N[i-1]))
		r6.append(-ln(e6[i]/e6[i-1])/ln(N[i]/N[i-1]))
		r7.append(-ln(e7[i]/e7[i-1])/ln(N[i]/N[i-1]))
		r8.append(-ln(e8[i]/e8[i-1])/ln(N[i]/N[i-1]))
		r9.append(-ln(e9[i]/e9[i-1])/ln(N[i]/N[i-1]))
		r10.append(-ln(e10[i]/e10[i-1])/ln(N[i]/N[i-1]))
		r11.append(-ln(e11[i]/e11[i-1])/ln(N[i]/N[i-1]))

	# Output LaTeX table
	if latexflag:
		outputTableLatex([e1,e2,e3,e4,e5,e6], [r1,r2,r3,r4,r5,r6], N[0], int(N[1]/N[0]), ltxname1)
		outputTableLatex([e7,e8,e9,e10,e11,e6], [r7,r8,r9,r10,r11,r6], N[0], int(N[1]/N[0]), ltxname2)
	
	# Print results		
	if printflag:
		print "Velocity error"
		for i in range (0,len(e1)):
			print "%10.2E %.2f" % (e1[i],r1[i])

		print "Pressure error"
		for i in range (0,len(e1)):
			print "%10.2E %.2f" % (e2[i],r2[i])

		print "Displacement error"
		for i in range (0,len(e1)):
			print "%10.2E %.2f" % (e3[i],r3[i])

		print "Stress L2 error"
		for i in range (0,len(e1)):
			print "%10.2E %.2f" % (e4[i],r4[i])

		print "div stress L2 error"
		for i in range (0,len(e1)):
			print "%10.2E %.2f" % (e5[i],r5[i])

		print "Rotation error"
		for i in range (0,len(e1)):
			print "%10.2E %.2f" % (e6[i],r6[i])
	

