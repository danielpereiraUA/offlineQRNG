clearvars
close all
clc

A=csvread('dataTest3.csv',2,0);
time = A(:,1)*1e-3;
amplitude = A(:,2)*1e-3;

% Gray code generator
nGray = 8;
bins = 2^nGray;
grayCode = ["0", "1"];
for i=1:nGray-1
    a=strings(1,2^i);
    b=strings(1,2^i);
    for j=1:length(a)
        a(j)="0";
        b(j)="1";
    end
    aux=fliplr(grayCode);
    aux=join([b ; aux]')';
    grayCode=join([a ; grayCode]')';
    grayCode=[grayCode, aux]; %#ok<AGROW>
end
grayCode=char(grayCode);
clear aux a b
for i=1:2^nGray
    aux=grayCode(:,:,i);
    aux(aux==' ')=[];
    GrayCode(:,:,i)=aux;
end
grayCode=GrayCode;
clear GrayCode

width = (100e-3)/2^8;
binLimits=zeros(1,bins);

binLimits(1)=-50e-3+width/2;

for i=2:bins
    binLimits(i)=binLimits(i-1)+width;
end

hist(amplitude,binLimits)
[counts]=hist(amplitude,binLimits);
probability = counts/length(amplitude);
