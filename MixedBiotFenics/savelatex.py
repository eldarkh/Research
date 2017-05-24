def outputTableLatex(errors, rates, hmax, hstep, name):
	'''Function to output errors and rates in a table written in LaTeX format'''
	numRows = len(errors[0])
	h = [hmax*(2**i) for i in range(numRows)]
	# Formatting
	sciF = '%4.2e'
	floatF = '%4.2f'

	# Open file
	dest = open(name, 'w')

	# Headers and begins
	dest.write('\\begin{table}[ht!]\n')
	dest.write('\\caption{Mixed Finite Element method for Biot poroelastic system.} \\label{tab:1}\n')
	dest.write('\\begin{center}\n')
	dest.write('\\begin{tabular}{c|cc|cc|cc}\n')

	# First level of table
	dest.write('\\hline\n')
	dest.write('& \\multicolumn{2}{c|}{$\|z - z_h\|_{L^2(\Omega)}$} & \multicolumn{2}{c|}{$\|p -p_h\|_{L^2(\Omega)}$} & \multicolumn{2}{c}{$\|u - u_h\|_{L^2(\Omega)}$} \\\ \n')
	dest.write('h & error & order & error & order & error & order \\\ \n')
	dest.write('\\hline\n')
	
	# Output data
	for i in range(numRows):
		dest.write('1/'+str(h[i]) + ' & ' + str(sciF % errors[0][i]) + ' & ' + str(floatF % rates[0][i]) \
								  + ' & ' + str(sciF % errors[1][i]) + ' & ' + str(floatF % rates[1][i]) \
								  + ' & ' + str(sciF % errors[2][i]) + ' & ' + str(floatF % rates[2][i]) + ' \\\ \n')

	# Second level of table
	dest.write('\\hline\n')
	dest.write('& \\multicolumn{2}{c|}{$\|\sigma - \sigma_h\|_{L^2(\Omega)}$} & \multicolumn{2}{c|}{$\|\\nabla \cdot (\sigma -\sigma_h)\|_{L^2(\Omega)}$} & \multicolumn{2}{c}{$\|\gamma - \gamma_h\|_{L^2(\Omega)}$} \\\ \n')
	dest.write('h & error & order & error & order & error & order \\\ \n')
	dest.write('\\hline\n')
	
	# Output data
	for i in range(numRows):
		dest.write('1/'+str(h[i]) + ' & ' + str(sciF % errors[3][i]) + ' & ' + str(floatF % rates[3][i]) \
								  + ' & ' + str(sciF % errors[4][i]) + ' & ' + str(floatF % rates[4][i]) \
								  + ' & ' + str(sciF % errors[5][i]) + ' & ' + str(floatF % rates[5][i]) + ' \\\ \n')

	# Close latex environments
	dest.write('\\hline\n')
	dest.write('\\end{tabular}\n')
	dest.write('\\end{center}\n')
	dest.write('\\end{table}\n')   
	dest.close()

# errors = [[1,2,3,4,5],[11,12,13,14,15],[21,22,23,24,25],[31,32,33,34,35],[41,42,42,44,45],[51,52,53,54,55]]
# rates = [[7,8,9,10,11],[17,18,19,110,111],[27,28,29,210,211],[37,38,39,310,311],[47,48,49,410,411],[57,58,59,510,511]]
# hmax = 2
# hstep = 2
# name = 'testfile.txt'
# outputTableLatex(errors, rates, hmax, hstep, name)

# outputTableLatex(errors, rates, hmax, hstep, name)
