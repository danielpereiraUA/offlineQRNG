import numpy as np
import hashlib as hash



with open('out.csv') as f:
    content = f.readlines()

binary = [x.strip() for x in content]

nGray = len(binary[0])
grayCode = ["0", "1"]
aux1 = ["1", "0"]
for i in range(nGray-1):
    aux0 = grayCode + []
    aux1 = aux0 + []
    aux1.reverse()
    for j in range(0, len(aux0)):
        aux0[j] = '0'+aux0[j]
        aux1[j] = '1'+aux1[j]
    grayCode = aux0+aux1
    aux0 = grayCode+[]

concatBinary = ''
counts = np.zeros(len(grayCode))
for i in range(len(binary)):
    counts[grayCode.index(binary[i])] = counts[grayCode.index(binary[i])] + 1
    concatBinary = concatBinary+binary[i]

probability = counts / len(binary)

entropyTotal = -np.sum(probability*np.log2(probability))

L = len(concatBinary)
ascii = []

i = 0
while i < L-8:
    if concatBinary[i] == '0':
        auxL = 8
        asciiByte = (concatBinary[i:i+auxL])
    else:
        auxL = 7
        asciiByte = ('0'+concatBinary[i:i+auxL])
    i = i+auxL
    aux = int(asciiByte, 2)
    ascii.append(aux.to_bytes((aux.bit_length() + 7) // 8, 'big').decode())

hashingFactor = 1365
preHashing = []
aux = ''
for i in range(len(ascii)):
    aux = aux + ascii[i]
    if len(aux) == hashingFactor:
        preHashing.append(aux)
        aux = ''
        print(preHashing[0])
        print(preHashing)
        input()

m = hash.sha512()

print(preHashing[0])

M = []
for i in range(len(preHashing)):
    M.append(hash.sha512(preHashing[i].encode()).hexdigest())

print(M[0])
print(preHashing[0])
