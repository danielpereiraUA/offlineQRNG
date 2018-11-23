import numpy as np
import hashlib as hash
import csv


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

hashingFactor = 11000
preHashing = []
aux = ''
for i in range(len(ascii)):
    aux = aux + ascii[i]
    if len(aux) == hashingFactor:
        preHashing.append(aux)
        aux = ''

m = hash.sha512()

M = []

finalString = ''

scale = 16  # equals to hexadecimal
num_of_bits = 8

for i in range(len(preHashing)):
    print(preHashing[i].encode())
    M.append(hash.sha512(preHashing[i].encode()).hexdigest())
    finalString = finalString + bin(int(M[i], scale))[2:].zfill(num_of_bits)

with open('finalOutSmaller.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, lineterminator='\n')
    writer.writerow([finalString])


print(len(finalString))
print(len(preHashing)*512)
