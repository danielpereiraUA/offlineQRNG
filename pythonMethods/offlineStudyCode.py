import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfinv
import csv

A=np.loadtxt(open("../csvData/dataTest.csv", "rb"), delimiter=",", skiprows=4)
time=A[:,0]*1e-3        # converting to s
amplitude=A[0:len(A):2,1]*1e-3   # converting to V
meanAmplitude = np.mean(amplitude)
stdAmplitude=np.std(amplitude)

# Gray code generator
nGray = 3;
grayCode = ["0","1"]
aux1 = ["1","0"]
for i in range(nGray-1):
    aux0 = grayCode + []
    aux1 = aux0 + []
    aux1.reverse()
    for j in range(0,len(aux0)):
        aux0[j]='0'+aux0[j]
        aux1[j]='1'+aux1[j]
    grayCode=aux0+aux1
    aux0=grayCode+[]

grayCodeValue=np.zeros([2**nGray]);
for i in range(2**nGray):
    for j in range(nGray):
        grayCodeValue[i]=grayCodeValue[i]+2*int(grayCode[i][j])-1

# Bin limit calculator
binLimits = np.zeros([2**nGray-1])
N=2**(nGray-1)
if nGray==1:
    binLimits = meanAmplitude
else:
    for i in range(N):
        binLimits[N-i-1] = meanAmplitude + np.sqrt(2)*stdAmplitude*erfinv(-float(i)/N)
        if i!=0:
            binLimits[N+i-1] = meanAmplitude + np.sqrt(2)*stdAmplitude*erfinv(float(i)/N)
#
# outBinary=[ ]
# for i in range(len(amplitude)):
#     aux=amplitude[i]<binLimits
#     outBinary.append(grayCode[len(grayCode)-sum(aux)-1])
#
# OutBinary=''.join(outBinary)
# with open('out.txt','w') as outFile:
#     print(OutBinary, file=outFile)

#print(outBinary)

#for i=1:nGray-1
#    a=strings(1,2^i);
#    b=strings(1,2^i);
#    for j=1:length(a)
#        a(j)="0";
#        b(j)="1";
#    end
#    aux=fliplr(grayCode);
#    aux=join([b ; aux]')';
#    grayCode=join([a ; grayCode]')';
#    grayCode=[grayCode, aux]; %#ok<AGROW>
#end
#grayCode=char(grayCode);
#clear aux a b
#for i=1:2^nGray
#    aux=grayCode(:,:,i);
#    aux(aux==' ')=[];
#    GrayCode(:,:,i)=aux;
#end
#grayCode=GrayCode;
#clear GrayCode
