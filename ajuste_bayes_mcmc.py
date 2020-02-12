import numpy as np 
import matplotlib.pyplot as plt 

data = np.genfromtxt('notas_andes.dat')

fisi_1 = data[:,0]
fisi_2 = data[:,1]
algebra = data[:,2]
diferencial = data[:,3]
global_pga = data[:,4]

datas = [fisi_1,fisi_2,algebra,diferencial,global_pga]

def log_likelihood(data,b):
	chi_cuad = 0

	for f1,f2,a,d,p in zip(data[0],data[1],data[2],data[3],data[4]):
		chi_cuad += (((b[0] + b[1]*f1 + b[2]*f2 + b[3]*a + b[4]*d)-p)**2)/(0.1**2)

	return (-chi_cuad/2.)

def MC(observados,pasos):
	betas = [0.1,0.1,0.1,0.1,0.1]
	L_parado = log_likelihood(observados,betas)
	betas_camino = betas

	for i in range(0,pasos):
		betas_ale = np.random.normal(betas, 0.01, size = 5)
		L_ale = log_likelihood(observados,betas_ale)
		alpha = np.exp(L_ale-L_parado)

		if alpha >= 1:
			betas_camino = np.vstack([betas_camino,betas_ale])
			betas = betas_ale
			L_parado = L_ale

		else:
			beta = np.random.random()

			if alpha >= beta:
				betas_camino = np.vstack([betas_camino,betas_ale])
				betas = betas_ale
				L_parado = L_ale

			else:
				betas_camino = np.vstack([betas_camino,betas])

	return betas_camino

betas = MC(datas,20000)

plt.figure(figsize = (9,10))
for i in range(0,5):
	plt.subplot(3,2,i+1)
	plt.hist(betas[10000:,i], bins = 20, density = True)
	plt.xlabel(r'$\beta_{}$'.format(i))
	plt.title(r'$\beta_{} = {:.2f} \pm {:.2f}$'.format(i,np.mean(betas[10000:,i]),
		np.std(betas[10000:,i])))

plt.subplots_adjust(hspace=0.6)
plt.savefig('ajuste_bayes_mcmc.png')