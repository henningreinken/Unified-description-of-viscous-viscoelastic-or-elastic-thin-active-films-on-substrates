
import sys
import numpy as np
import os

# get current directory
thisPath = os.getcwd()

nuPStart = 12.0
nuPEnd = 24.0
nuPStep = 0.2

numnuP = int(np.round((nuPEnd - nuPStart)/nuPStep + 1))

nuPArray = np.linspace(nuPStart,nuPEnd,numnuP,endpoint=True)

oscDataFile = open(thisPath+'/dataFourierAnalysis.dat','w')

# walk through all the values
for nuP in nuPArray:
	
	print(nuP)
	
	# folder name
	jobName = 'nuP%08.3f'%(nuP)
	jobName = jobName.replace('.','d',1)
	folderName = thisPath + "/" + jobName

	# get data 
	dataP = np.genfromtxt(folderName+'/dataPMean.dat')
	datav = np.genfromtxt(folderName+'/datavMean.dat')
	datau = np.genfromtxt(folderName+'/datauMean.dat')
	
	# extract time step
	dt = round(dataP[1,0] - dataP[0,0],6)

	# number of time steps
	NP = np.shape(dataP)[0]
	Nv = np.shape(datav)[0]
	Nu = np.shape(datau)[0]

	# use less data for analysis
	NStartP = int(NP/2)
	NStartv = int(Nv/2)
	NStartu = int(Nu/2)
	
	# mean absolute values of fields
	PAbsMean = np.mean(np.sqrt(dataP[0:int(NP/2),1]**2 + dataP[0:int(NP/2):,2]**2))
	vAbsMean = np.mean(np.sqrt(datav[0:int(Nv/2):,1]**2 + datav[0:int(Nv/2):,2]**2))
	uAbsMean = np.mean(np.sqrt(datau[0:int(Nu/2):,1]**2 + datau[0:int(Nu/2):,2]**2))

	# for zero padding
	NFP = 10*(NP-NStartP)
	NFv = 10*(Nv-NStartv)
	NFu = 10*(Nu-NStartu)
	
	# Fourier transform
	dataPF = np.fft.fft(dataP[NStartP:,:],n=NFP,axis=0)
	datavF = np.fft.fft(datav[NStartv:,:],n=NFv,axis=0)
	datauF = np.fft.fft(datau[NStartu:,:],n=NFu,axis=0)
	
	# frequencies
	freqP = np.fft.fftfreq(NFP,dt)
	freqv = np.fft.fftfreq(NFv,dt)
	frequ = np.fft.fftfreq(NFu,dt)

	# determine dominating frequency
	freqPxMaxInd = np.argmax(np.absolute(dataPF[:int(NFP/2),1]))
	freqPxMax = abs(freqP[freqPxMaxInd])
	freqPyMaxInd = np.argmax(np.absolute(dataPF[:int(NFP/2),2]))
	freqPyMax = abs(freqP[freqPyMaxInd])

	freqvxMaxInd = np.argmax(np.absolute(datavF[:int(NFv/2),1]))
	freqvxMax = abs(freqv[freqvxMaxInd])
	freqvyMaxInd = np.argmax(np.absolute(datavF[:int(NFv/2),2]))
	freqvyMax = abs(freqv[freqvyMaxInd])

	frequxMaxInd = np.argmax(np.absolute(datauF[:int(NFu/2),1]))
	frequxMax = abs(freqP[frequxMaxInd])
	frequyMaxInd = np.argmax(np.absolute(datauF[:int(NFu/2),2]))
	frequyMax = abs(freqP[frequyMaxInd])
	
	# determine phase shifts between quantities
	phasePx = np.angle(dataPF[freqPxMaxInd,1])
	phasevx = np.angle(datavF[freqvxMaxInd,1])
	phaseux = np.angle(datauF[frequxMaxInd,1])
	phaseShiftvx = phasevx - phasePx 
	phaseShiftux = phaseux - phasePx

	phasePy = np.angle(dataPF[freqPyMaxInd,2])
	phasevy = np.angle(datavF[freqvyMaxInd,2])
	phaseuy = np.angle(datauF[frequyMaxInd,2])
	phaseShiftvy = phasevy - phasePy 
	phaseShiftuy = phaseuy - phasePy
	
	# average frequencies and phase shifts over x and y
	freqPTransMax = 0.5*(freqPxMax + freqPyMax)
	freqvTransMax = 0.5*(freqvxMax + freqvyMax)
	frequTransMax = 0.5*(frequxMax + frequyMax)
	
	phaseShiftvTrans = 0.5*(phaseShiftvx + phaseShiftvy)
	phaseShiftuTrans = 0.5*(phaseShiftux + phaseShiftuy)
	
	# write result of Fourier analysis to data file
	oscDataFile.write(str(nuP)+' ')
	oscDataFile.write(str(PAbsMean)+' ') 
	oscDataFile.write(str(vAbsMean)+' ') 
	oscDataFile.write(str(uAbsMean)+' ') 
	
	# if there are no global rotations, write nan
	if PAbsMean < 0.0001:
		freqPTransMax = np.nan
		freqvTransMax = np.nan
		frequTransMax = np.nan
		phaseShiftvTrans = np.nan
		phaseShiftuTrans = np.nan
	
	oscDataFile.write(str(2.0*np.pi*freqPTransMax)+' ') 
	oscDataFile.write(str(2.0*np.pi*freqvTransMax)+' ') 
	oscDataFile.write(str(2.0*np.pi*frequTransMax)+' ') 
	
	oscDataFile.write(str(phaseShiftvTrans)+' ')
	oscDataFile.write(str(phaseShiftuTrans)+' ')
	
	oscDataFile.write('\n')
	
oscDataFile.close()
