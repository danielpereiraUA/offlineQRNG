#include <stdio.h>

#include <math.h>
#include <string.h>
#include <stdlib.h>


#include "matrix.h"
#include "defs.h"
#include "fft.h"
#include "cephes.h"
#include "externs.h"

int FileExists(const char *filename);

int Frequency(unsigned long int *binaryString, int size);
int BlockFrequency(unsigned long int *binaryString, int size, int M);
int Runs(unsigned long int *epsilon, int n);
int LongestRunOfOnes(unsigned long int *epsilon, int n);
int Rank(unsigned long int *epsilon, int n);
int DiscreteFourierTransform(unsigned long int *epsilon, int n);
int LinearComplexity(unsigned long int *epsilon, int M, int n);
int CumulativeSums(unsigned long int *epsilon, int n);
double RandomExcursions(unsigned long int *epsilon, int n);
double RandomExcursionsVariant(unsigned long int *epsilon, int n);
int Universal(unsigned long int *epsilon, int n);
int Serial(unsigned long int *epsilon, int m, int n);
int ApproximateEntropy(unsigned long int *epsilon, int m, int n);
double NonOverlappingTemplateMatchings(unsigned long int *epsilon, int m, int n);
int OverlappingTemplateMatchings(unsigned long int *epsilon, int m, int n);

#define buffer_size 579986

int main() {
	//int argc, char* argv[]) {

	
//	const int buffer_size = atoi(argv[1]);

	unsigned long int * ptr;
	ptr = (unsigned long int *)malloc((buffer_size >> auxSize) * sizeof(ptr));
	memset(ptr, 0, (buffer_size >> auxSize) * sizeof(ptr));
	
	char aux;

	errno_t err;

	FILE *inFile;
	err = fopen_s( &inFile, "finalOut.csv", "r+");
	int i;
	for (i = 0; i < buffer_size; i++)
	{
		fscanf_s(inFile, "%c", &aux, sizeof(char));
		ptr[i >> auxSize] <<= 1;
		if (aux == '1') ptr[i >> auxSize] |= 1;
	}
	fclose(inFile);

	printf_s("hello! \n");

	int freqResult = Frequency(ptr, buffer_size);
	int freqBlockResult = BlockFrequency(ptr, buffer_size, 128);
	int runsResult = Runs(ptr, buffer_size);
	int runOfOnesResult = LongestRunOfOnes(ptr, buffer_size);
	int rankResult = Rank(ptr, buffer_size);
	int fftResult = DiscreteFourierTransform(ptr, buffer_size);
	int linearComplexityResult = LinearComplexity(ptr, 500, buffer_size);
	int cumSumResult = CumulativeSums(ptr, buffer_size);
	double randExcursionResult = RandomExcursions(ptr, buffer_size);
	double randExcursionVariantResult = RandomExcursionsVariant(ptr, buffer_size);
	int universalResult = Universal(ptr, buffer_size);
	int serialResult = Serial(ptr, 2, buffer_size);
	int entropyResult = ApproximateEntropy(ptr, 10, buffer_size);
	double nonOverTempResult = NonOverlappingTemplateMatchings(ptr, 9, buffer_size);
	int overTempResult = OverlappingTemplateMatchings(ptr, 9, buffer_size);

	printf("Frequency test %i \n", freqResult);
	printf("Block frequency test %i \n", freqBlockResult);
	printf("Runs test %i \n", runsResult);
	printf("Run Of Ones test %i \n", runOfOnesResult);
	printf("Matrix rank test %i \n", rankResult);
	printf("Fourier transform test %i \n", fftResult);
	printf("Linear complexity test %i \n", linearComplexityResult);
	printf("Cumsum test %i \n", linearComplexityResult);
	printf("Random excursion test %lf \n", randExcursionResult);
	printf("Random excursion variant test %lf \n", randExcursionVariantResult);
	printf("Universal statistical test %i \n", universalResult);
	printf("Serial test %i \n", serialResult);
	printf("Aproximate entropy test %i \n", entropyResult);
	printf("Non Overlapping Template test %lf \n", nonOverTempResult);
	printf("Overlapping Template test %i \n", overTempResult);
	
	/*double fullyRandom = freqResult + freqBlockResult + runsResult + runOfOnesResult + rankResult + fftResult
		+ linearComplexityResult + cumSumResult + randExcursionResult + randExcursionVariantResult + universalResult
		+ serialResult + entropyResult + nonOverTempResult + overTempResult;

	if (freqResult && freqBlockResult && runsResult && runOfOnesResult && rankResult && fftResult && linearComplexityResult && cumSumResult && randExcursionResult && randExcursionVariantResult && universalResult && serialResult && entropyResult && nonOverTempResult && overTempResult)
	{
		if (fullyRandom == 15)
		{
			// All tests were fully cleared
			if (FileExists("confirmedRandom.txt"))
			{
				// confirmedRandom.txt already exists, new string is appended to it
				FILE *outFile;
				err = fopen_s(&outFile, "confirmedRandom.txt", "a");

				unsigned long int msk = 1 << 31;
				int i;
				for (i = 0; i < buffer_size; i++)
				{
					if (msk & ptr[i >> auxSize])
					{
						// it's a one
						fwrite("1", 1, sizeof("1"), outFile);
					}
					else
					{
						// it's a zero
						fwrite("0", 1, sizeof("1"), outFile);
					}
					msk >>= 1;
					if (msk == 0) msk = 1 << 31;
				}

				fclose(outFile);

			}
			else
			{
				// confirmedRandom.txt doesn't exist, testingBinary.txt is renamed to confirmedRandom.txt
				rename("testingBinary.txt", "confirmedRandom.txt");
			}

		}
		else
		{
			// All tests were cleared
			if (FileExists("confirmedRandom.txt"))
			{
				// inconclusiveRandom.txt already exists, new string is appended to it
				FILE *outFile;
				err = fopen_s(&outFile, "inconclusiveRandom.txt", "a");

				unsigned long int msk = 1 << 31;
				int i;
				for (i = 0; i < buffer_size; i++)
				{
					if (msk & ptr[i >> auxSize])
					{
						// it's a one
						fwrite("1", 1, sizeof("1"), outFile);
					}
					else
					{
						// it's a zero
						fwrite("0", 1, sizeof("1"), outFile);
					}
					msk >>= 1;
					if (msk == 0) msk = 1 << 31;
				}

				fclose(outFile);

			}
			else
			{
				// inconclusiveRandom.txt doesn't exist, testingBinary.txt is renamed to inconclusiveRandom.txt
				rename("testingBinary.txt", "inconclusiveRandom.txt");
			}
		}
	}
	else
	{
		remove("testingBinary.txt");
	}
	*/
	free(ptr);

	printf("end!");
	getchar();
	return 0;
	
}

int FileExists(const char *filename)
{
	FILE *fp = fopen(filename, "r");
	if (fp != NULL) fclose(fp);
	return (fp != NULL); 
}


/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
						  F R E Q U E N C Y  T E S T        CORRECT
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

int Frequency(unsigned long int *epsilon, int n) {
	double	f, s_obs, p_value, sum, sqrt2 = 1.41421356237309504880;

	sum = 0.0;

	unsigned long int msk = 1 << 31;
	int i;
	for (i = 0; i < n; i++)
	{
		if (msk & epsilon[i >> auxSize])
		{
			// it's a one
			sum += 2 - 1;
		}
		else
		{
			// it's a zero
			sum += -1;
		}
		msk >>= 1;
		if (msk == 0) msk = 1 << 31;
	}

	s_obs = fabs(sum) / sqrt(n);
	f = s_obs / sqrt2;
	p_value = erfc(f);

	if (p_value > 0.01)
	{
		return 1;
	}
	else
	{
		return 0;
	}
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
					B L O C K  F R E Q U E N C Y  T E S T    CORRECT
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

int BlockFrequency(unsigned long int *binaryString, int size, int M) {
	int		i, j, N, blockSum;
	double	p_value, sum, pi, v, chi_squared;

	N = size / M; 		/* # OF SUBSTRING BLOCKS      */
	sum = 0.0;

	unsigned long int msk = 1 << 31;
	for (i = 0; i < N; i++) {
		blockSum = 0;
		for (j = 0; j < M; j++)
		{
			if (msk & binaryString[(j + i * M) >> auxSize])
			{
				// it's a one
				blockSum++;
			}
			else
			{
				// it's a zero
			}

			msk >>= 1;
			if (msk == 0) msk = 1 << 31;
		}

			//blockSum += binaryString[j + i * M];
		pi = (double)blockSum / (double)M;
		v = pi - 0.5;
		sum += v * v;
	}
	chi_squared = 4.0 * M * sum;
	p_value = cephes_igamc( (double)N / 2.0, chi_squared / 2.0);

	if (p_value > 0.01)
	{
		return 1;
	}
	else
	{
		return 0;
	}
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
							  R U N S  T E S T              CORRECT
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

int Runs(unsigned long int *epsilon, int n) {
	int		S, k;
	double	pi, V, erfc_arg, p_value;
	unsigned long int msk = 1 << 31;

	S = 0;
	for (k = 0; k < n; k++)
	{
		if (msk & epsilon[k >> auxSize]) S++;
		msk >>= 1;
		if (msk == 0) msk = 1 << 31;
	}
	pi = (double)S / (double)n;

	 msk = 1 << 31;
	int auxRunsTestBefore = 0;
	int auxRunsTestNow = 0;
	if (msk & epsilon[0]) auxRunsTestBefore = 1;

	msk >>= 1;

	if (fabs(pi - 0.5) > (2.0 / sqrt(n))) {
		return 0;
	}
	else {

		V = 1;
		for (k = 1; k < n; k++)
		{
			auxRunsTestNow = 0;
			if (msk & epsilon[k >> auxSize]) auxRunsTestNow = 1;
				
			if (auxRunsTestNow != auxRunsTestBefore) V++;

			auxRunsTestBefore = auxRunsTestNow;

			msk >>= 1;
			if (msk == 0) msk = 1 << 31;
		}
		erfc_arg = fabs(V - 2.0 * n * pi * (1 - pi)) / (2.0 * pi * (1 - pi) * sqrt(2 * n));
		p_value = erfc(erfc_arg);
	}

	if (p_value > 0.01)
	{
		return 1;
	}
	else
	{
		return 0;
	}
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
					  L O N G E S T  R U N S  T E S T		CORRECT
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

int LongestRunOfOnes(unsigned long int *epsilon, int n) {
	double			pval, chi2, pi[7];
	int				run, v_n_obs, N, i, j, K, M, V[7];
	unsigned int	nu[7] = { 0, 0, 0, 0, 0, 0, 0 };
	unsigned long int msk = 1 << 31;

	// This code populates pi
	if (n < 128) {
		return 0;
	}
	if (n < 6272) {
		K = 3;
		M = 8;
		V[0] = 1; V[1] = 2; V[2] = 3; V[3] = 4;
		pi[0] = 0.21484375;
		pi[1] = 0.3671875;
		pi[2] = 0.23046875;
		pi[3] = 0.1875;
	}
	else if (n < 750000) {
		K = 5;
		M = 128;
		V[0] = 4; V[1] = 5; V[2] = 6; V[3] = 7; V[4] = 8; V[5] = 9;
		pi[0] = 0.1174035788;
		pi[1] = 0.242955959;
		pi[2] = 0.249363483;
		pi[3] = 0.17517706;
		pi[4] = 0.102701071;
		pi[5] = 0.112398847;
	}
	else {
		K = 6;
		M = 10000;
		V[0] = 10; V[1] = 11; V[2] = 12; V[3] = 13; V[4] = 14; V[5] = 15; V[6] = 16;
		pi[0] = 0.0882;
		pi[1] = 0.2092;
		pi[2] = 0.2483;
		pi[3] = 0.1933;
		pi[4] = 0.1208;
		pi[5] = 0.0675;
		pi[6] = 0.0727;
	}


	N = n / M;
	for (i = 0; i < N; i++) {
		v_n_obs = 0;
		run = 0;
		for (j = 0; j < M; j++) {
			if (msk & epsilon[(i*M + j) >> auxSize]) {
				run++;
				if (run > v_n_obs)
					v_n_obs = run;
			}
			else
				run = 0;
			msk >>= 1;
			if (msk == 0) msk = 1 << 31;

		}



		if (v_n_obs < V[0])
			nu[0]++;
		for (j = 0; j <= K; j++) {
			if (v_n_obs == V[j])
				nu[j]++;
		}
		if (v_n_obs > V[K])
			nu[K]++;
	}

	chi2 = 0.0;
	for (i = 0; i <= K; i++)
		chi2 += ((nu[i] - N * pi[i]) * (nu[i] - N * pi[i])) / (N * pi[i]);

	pval = cephes_igamc((double)(K / 2.0), chi2 / 2.0);

	if (pval > 0.01)
	{
		return 1;
	}
	else
	{
		return 0;
	}
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
							  R A N K  T E S T				CORRECT
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

int Rank(unsigned long int *epsilon, int n) {
	int			N, i, k, r;
	double		p_value, product, chi_squared, arg1, p_32, p_31, p_30, R, F_32, F_31, F_30;
	unsigned long int msk = 1 << 31;
	BitSequence	**matrix = create_matrix(32, 32);

	N = n / (32 * 32);
	if (isZero(N)) {
		p_value = 0.00;
	}
	else {
		r = 32;					// COMPUTE PROBABILITIES 
		product = 1;
		for (i = 0; i <= r - 1; i++)
			product *= ((1.e0 - pow(2, i - 32))*(1.e0 - pow(2, i - 32))) / (1.e0 - pow(2, i - r));
		p_32 = pow(2, r*(32 + 32 - r) - 32 * 32) * product;

		r = 31;
		product = 1;
		for (i = 0; i <= r - 1; i++)
			product *= ((1.e0 - pow(2, i - 32))*(1.e0 - pow(2, i - 32))) / (1.e0 - pow(2, i - r));
		p_31 = pow(2, r*(32 + 32 - r) - 32 * 32) * product;

		p_30 = 1 - (p_32 + p_31);

		F_32 = 0;
		F_31 = 0;
		for (k = 0; k < N; k++) {			// FOR EACH 32x32 MATRIX   

			int M = 32, Q = 32;

			int		i, j;

			for (i = 0; i < M; i++)
			{
				for (j = 0; j < Q; j++)
				{
					if (msk & epsilon[(k*(M*Q) + j + i * M) >> auxSize])
					{
						matrix[i][j] = 1;
					}
					else matrix[i][j] = 0;
					msk >>= 1;
					if (msk == 0) msk = 1 << 31;
				}

			}
 
 			R = computeRank(32, 32, matrix);
			if (R == 32)
				F_32++;			// DETERMINE FREQUENCIES 
			if (R == 31)
				F_31++;
		}
		F_30 = (double)N - (F_32 + F_31);

		chi_squared = (pow(F_32 - N * p_32, 2) / (double)(N*p_32) +
			pow(F_31 - N * p_31, 2) / (double)(N*p_31) +
			pow(F_30 - N * p_30, 2) / (double)(N*p_30));

		arg1 = -chi_squared / 2.e0;

		p_value = exp(arg1);

		for (i = 0; i < 32; i++)				// DEALLOCATE MATRIX  
			free(matrix[i]);
		free(matrix);
	}

	if (p_value > 0.01)
	{
		return 1;
	}
	else
	{
		return 0;
	}

}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
		 D I S C R E T E  F O U R I E R  T R A N S F O R M  T E S T CORRECT
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

int DiscreteFourierTransform(unsigned long int *epsilon, int n) {
	double	p_value, upperBound, percentile, N_l, N_o, d, *m, *X, *Y;
	int		i, count;

	m = (double *)malloc(n / 2 * sizeof(double));
	X = (double *)malloc(n * sizeof(double));
	Y = (double *)malloc(n * sizeof(double));

	unsigned long int msk = 1 << 31;
	int k;
	for (k = 0; k < n; k++)
	{
		if (msk & epsilon[k >> auxSize])
		{
			// it's a one
			X[k] = 1;
		}
		else
		{
			// it's a zero
			X[k] = -1;
		}
		Y[k] = 0;
		msk >>= 1;
		if (msk == 0) msk = 1 << 31;
	}

	Fft_transform(X, Y, n);

	// COMPUTE MAGNITUDE 
	for (i = 0; i < n/2 ; i++)
		m[i] = sqrt(pow(X[i], 2) + pow(Y[i], 2));

	count = 0;				       // CONFIDENCE INTERVAL 
	upperBound = sqrt(2.995732274*n);

	for (i = 0; i < n/2 ; i++)
		if (m[i] < upperBound)
			count++;
	percentile = (double)count / (n / 2) * 100;
	N_l = (double)count;       // number of peaks less than upperBound
	N_o = (double) 0.95*n / 2.0; //number of peaks expected to be be less than upperBound
	d = (N_l - N_o) / sqrt(n / 4.0*0.95*0.05);
	p_value = erfc(fabs(d) / sqrt(2.0));

	free(m);
	free(X);
	free(Y);

	if (p_value > 0.01)
	{
		return 1;
	}
	else
	{
		return 0;
	}
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
			L I N E A R  C O M P L E X I T Y  T E S T    CORRECT
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

int LinearComplexity(unsigned long int *epsilon, int M, int n) {
	int       i, ii, j, d, N, L, m, N_, parity, sign, K = 6;
	double    p_value, T_, mean, nu[7], chi2;
	double    pi[7] = { 0.01047, 0.03125, 0.12500, 0.50000, 0.25000, 0.06250, 0.020833 };
	BitSequence  *T = NULL, *P = NULL, *B_ = NULL, *C = NULL;

	int *auxBinary;
	unsigned long int msk = 1 << 31;

	auxBinary = (int *)malloc(n * sizeof(int));
	int k;
	for (k = 0; k < n; k++)
	{
		if (msk & epsilon[k >> auxSize])
		{
			auxBinary[k] = 1;
		}
		else
		{
			auxBinary[k] = 0;
		}
		msk >>= 1;
		if (msk == 0) msk = 1 << 31;
	}


	

	N = (int)floor(n / M);
	if (((B_ = (BitSequence *)calloc(M, sizeof(BitSequence))) == NULL) ||
		((C = (BitSequence *)calloc(M, sizeof(BitSequence))) == NULL) ||
		((P = (BitSequence *)calloc(M, sizeof(BitSequence))) == NULL) ||
		((T = (BitSequence *)calloc(M, sizeof(BitSequence))) == NULL)) {
		if (B_ != NULL)
			free(B_);
		if (C != NULL)
			free(C);
		if (P != NULL)
			free(P);
		if (T != NULL)
			free(T);
		return 0;
	}

	for (i = 0; i < K + 1; i++)
		nu[i] = 0.00;
	for (ii = 0; ii < N; ii++) {
		for (i = 0; i < M; i++) {
			B_[i] = 0;
			C[i] = 0;
			T[i] = 0;
			P[i] = 0;
		}
		L = 0;
		m = -1;
		d = 0;
		C[0] = 1;
		B_[0] = 1;

		/* DETERMINE LINEAR COMPLEXITY */
		N_ = 0;
		while (N_ < M) {
			d = (int)auxBinary[ii*M + N_];

			for (i = 1; i <= L; i++)
				d += C[i] * auxBinary[ii*M + N_ - i];
			d = d % 2;
			if (d == 1) {
				for (i = 0; i < M; i++) {
					T[i] = C[i];
					P[i] = 0;
				}
				for (j = 0; j < M; j++)
					if (B_[j] == 1)
						P[j + N_ - m] = 1;
				for (i = 0; i < M; i++)
					C[i] = (C[i] + P[i]) % 2;
				if (L <= N_ / 2) {
					L = N_ + 1 - L;
					m = N_;
					for (i = 0; i < M; i++)
						B_[i] = T[i];
				}
			}
			N_++;
		}
		if ((parity = (M + 1) % 2) == 0)
			sign = -1;
		else
			sign = 1;
		mean = M / 2.0 + (9.0 + sign) / 36.0 - 1.0 / pow(2, M) * (M / 3.0 + 2.0 / 9.0);
		if ((parity = M % 2) == 0)
			sign = 1;
		else
			sign = -1;
		T_ = sign * (L - mean) + 2.0 / 9.0;

		if (T_ <= -2.5)
			nu[0]++;
		else if (T_ > -2.5 && T_ <= -1.5)
			nu[1]++;
		else if (T_ > -1.5 && T_ <= -0.5)
			nu[2]++;
		else if (T_ > -0.5 && T_ <= 0.5)
			nu[3]++;
		else if (T_ > 0.5 && T_ <= 1.5)
			nu[4]++;
		else if (T_ > 1.5 && T_ <= 2.5)
			nu[5]++;
		else
			nu[6]++;
	}
	chi2 = 0.00;
	for (i = 0; i < K + 1; i++)
		chi2 += pow(nu[i] - N * pi[i], 2) / (N*pi[i]);
	p_value = cephes_igamc(K / 2.0, chi2 / 2.0);

	free(B_);
	free(P);
	free(C);
	free(T);
	free(auxBinary);

	if (p_value > 0.01)
	{
		return 1;
	}
	else
	{
		return 0;
	}


}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
			C U M U L A T I V E  S U M S  T E S T             CORRECT
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

int CumulativeSums(unsigned long int *epsilon, int n) {
	int		S, sup, inf, z, zrev, k;
	double	sum1, sum2, p_value0, p_value1;
	unsigned long int msk = 1 << 31;

	S = 0;
	sup = 0;
	inf = 0;
	for (k = 0; k < n; k++) {
		if (msk & epsilon[k >> auxSize]) {
			S++;
		}
		else
		{
			S--;
		}
		//epsilon[k] ? S++ : S--;
		msk >>= 1;
		if (msk == 0) msk = 1 << 31;
		if (S > sup)
			sup++;
		if (S < inf)
			inf--;
		z = (sup > -inf) ? sup : -inf;
		zrev = (sup - S > S - inf) ? sup - S : S - inf;
	}

	// forward
	sum1 = 0.0;
	for (k = (-n / z + 1) / 4; k <= (n / z - 1) / 4; k++) {
		sum1 += cephes_normal(((4 * k + 1)*z) / sqrt(n));
		sum1 -= cephes_normal(((4 * k - 1)*z) / sqrt(n));
	}
	sum2 = 0.0;
	for (k = (-n / z - 3) / 4; k <= (n / z - 1) / 4; k++) {
		sum2 += cephes_normal(((4 * k + 3)*z) / sqrt(n));
		sum2 -= cephes_normal(((4 * k + 1)*z) / sqrt(n));
	}

	p_value0 = 1.0 - sum1 + sum2;

	// backwards
	sum1 = 0.0;
	for (k = (-n / zrev + 1) / 4; k <= (n / zrev - 1) / 4; k++) {
		sum1 += cephes_normal(((4 * k + 1)*zrev) / sqrt(n));
		sum1 -= cephes_normal(((4 * k - 1)*zrev) / sqrt(n));
	}
	sum2 = 0.0;
	for (k = (-n / zrev - 3) / 4; k <= (n / zrev - 1) / 4; k++) {
		sum2 += cephes_normal(((4 * k + 3)*zrev) / sqrt(n));
		sum2 -= cephes_normal(((4 * k + 1)*zrev) / sqrt(n));
	}
	p_value1 = 1.0 - sum1 + sum2;

	if (p_value0 > 0.01 && p_value1 > 0.01)
	{
		return 1;
	}
	else
	{
		return 0;
	}

}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
					 R A N D O M  E X C U R S I O N S  T E S T   CORRECT
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

double RandomExcursions(unsigned long int *epsilon, int n) {
	int		b, i, j, k, J, x, p_valueCounter = 0;
	int		cycleStart, cycleStop, *S_k, *cycle;

	S_k = (int *)malloc(n * sizeof(int));

	memset(S_k, 0, n * sizeof(int));

	cycle = (int *)malloc(MAX(1000, n / 100) * sizeof(int));

	int		stateX[8] = { -4, -3, -2, -1, 1, 2, 3, 4 };
	int		counter[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
	double	p_value, sum, constraint, nu[6][8];
	double	pi[5][6] = { {0.0000000000, 0.00000000000, 0.00000000000, 0.00000000000, 0.00000000000, 0.0000000000},
						 {0.5000000000, 0.25000000000, 0.12500000000, 0.06250000000, 0.03125000000, 0.0312500000},
						 {0.7500000000, 0.06250000000, 0.04687500000, 0.03515625000, 0.02636718750, 0.0791015625},
						 {0.8333333333, 0.02777777778, 0.02314814815, 0.01929012346, 0.01607510288, 0.0803755143},
						 {0.8750000000, 0.01562500000, 0.01367187500, 0.01196289063, 0.01046752930, 0.0732727051} };

	J = 0; 					/* DETERMINE CYCLES */


	unsigned long int msk = 1 << 31;
	if (msk & epsilon[0 >> auxSize]) S_k[0] = 1;
	else S_k[0] = -1;
	msk >>= 1;
	for (i = 1; i < n; i++) {
		if (msk & epsilon[i >> auxSize]) S_k[i] = S_k[i - 1] + 1;
		else S_k[i] = S_k[i - 1] - 1;

		msk >>= 1;
		if (msk == 0) msk = 1 << 31;

		if (S_k[i] == 0) {
			J++;
			if (J > MAX(1000, n / 100)) {
				free(S_k);
				free(cycle);
				return 0;
			}
			cycle[J] = i;
		}
	}

	if (S_k[n - 1] != 0)
		J++;
	cycle[J] = n;

	constraint = MAX(0.005*pow(n, 0.5), 500);
	if (J < constraint) {
		free(S_k);
		free(cycle);
		return 0;
	}
	else 
	{
		cycleStart = 0;
		cycleStop = cycle[1];
		for (k = 0; k < 6; k++)
			for (i = 0; i < 8; i++)
				nu[k][i] = 0.;
		for (j = 1; j <= J; j++) {                           /* FOR EACH CYCLE */
			for (i = 0; i < 8; i++)
				counter[i] = 0;
			for (i = cycleStart; i < cycleStop; i++) {
				if ((S_k[i] >= 1 && S_k[i] <= 4) || (S_k[i] >= -4 && S_k[i] <= -1)) {
					if (S_k[i] < 0)
						b = 4;
					else
						b = 3;
					counter[S_k[i] + b]++;
				}
			}
			cycleStart = cycle[j] + 1;
			if (j < J)
				cycleStop = cycle[j + 1];

			for (i = 0; i < 8; i++) {
				if ((counter[i] >= 0) && (counter[i] <= 4))
					nu[counter[i]][i]++;
				else if (counter[i] >= 5)
					nu[5][i]++;
			}
		}

		for (i = 0; i < 8; i++) {
			x = stateX[i];
			sum = 0.;
			for (k = 0; k < 6; k++)
				sum += pow(nu[k][i] - J * pi[(int)fabs(x)][k], 2) / (J*pi[(int)fabs(x)][k]);
			p_value = cephes_igamc(2.5, sum / 2.0);
			if (p_value > 0.01) p_valueCounter++;
		}
	}


	free(S_k);
	free(cycle);

	double result = (double)(p_valueCounter) / (double)(8);
	
	return result;
	
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
			R A N D O M  E X C U R S I O N S  V A R I A N T  T E S T CORRECT
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

double RandomExcursionsVariant(unsigned long int *epsilon, int n) {
	int		i, p, J, x, constraint, count, *S_k;
	int		stateX[18] = { -9, -8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	double	p_value;
	unsigned long int msk = 1 << 31;

	S_k = (int *)malloc(n * sizeof(int));

	J = 0;
	if (msk & epsilon[0 >> auxSize]) S_k[0] = 1;
	else S_k[0] = -1;
	msk >>= 1;

	for (i = 1; i < n; i++) {
		if (msk & epsilon[i >> auxSize]) S_k[i] = S_k[i - 1] + 1;
		else S_k[i] = S_k[i - 1] - 1;

		msk >>= 1;
		if (msk == 0) msk = 1 << 31;

		if (S_k[i] == 0)
			J++;
	}
	if (S_k[n - 1] != 0)
		J++;

	constraint = (int)MAX(0.005*pow(n, 0.5), 500);

	int p_valueCounter = 0;

	if (J < constraint) {
		free(S_k);
		return 0;
	}
	else {
		for (p = 0; p <= 17; p++) {
			x = stateX[p];
			count = 0;
			for (i = 0; i < n; i++)
				if (S_k[i] == x)
					count++;
			p_value = erfc(fabs(count - J) / (sqrt(2.0*J*(4.0*fabs(x) - 2))));
			if (p_value > 0.01) p_valueCounter++;
			
		}
	}
	free(S_k);

	double result = (double)(p_valueCounter) / (double)(18);

	return result;
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
						 U N I V E R S A L  T E S T                 CORRECT
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

int Universal(unsigned long int *epsilon, int n) {
	int		i, j, p, L, Q, K;
	double	arg, sqrt2, sigma, phi, sum, p_value, c;
	unsigned long int msk = 1 << 31;
	long	*T, decRep;
	double	expected_value[17] = { 0, 0, 0, 0, 0, 0, 5.2177052, 6.1962507, 7.1836656,
				8.1764248, 9.1723243, 10.170032, 11.168765,
				12.168070, 13.167693, 14.167488, 15.167379 };
	double   variance[17] = { 0, 0, 0, 0, 0, 0, 2.954, 3.125, 3.238, 3.311, 3.356, 3.384,
				3.401, 3.410, 3.416, 3.419, 3.421 };

	/* * * * * * * * * ** * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * THE FOLLOWING REDEFINES L, SHOULD THE CONDITION:     n >= 1010*2^L*L       *
	 * NOT BE MET, FOR THE BLOCK LENGTH L.                                        *
	 * * * * * * * * * * ** * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	L = 5;
	if (n >= 387840)     L = 6;
	if (n >= 904960)     L = 7;
	if (n >= 2068480)    L = 8;
	if (n >= 4654080)    L = 9;
	if (n >= 10342400)   L = 10;
	if (n >= 22753280)   L = 11;
	if (n >= 49643520)   L = 12;
	if (n >= 107560960)  L = 13;
	if (n >= 231669760)  L = 14;
	if (n >= 496435200)  L = 15;
	if (n >= 1059061760) L = 16;

	Q = 10 * (int)pow(2, L);
	K = (int)(floor(n / L) - (double)Q);	 		    /* BLOCKS TO TEST */

	p = (int)pow(2, L);
	if ((L < 6) || (L > 16) || ((double)Q < 10 * pow(2, L)) ||
		((T = (long *)calloc(p, sizeof(long))) == NULL)) {
		return 0;
	}

	/* COMPUTE THE EXPECTED:  Formula 16, in Marsaglia's Paper */
	c = 0.7 - 0.8 / (double)L + (4 + 32 / (double)L)*pow(K, -3 / (double)L) / 15;
	sigma = c * sqrt(variance[L] / (double)K);
	sqrt2 = sqrt(2);
	sum = 0.0;
	for (i = 0; i < p; i++)
		T[i] = 0;
	for (i = 1; i <= Q; i++) {		/* INITIALIZE TABLE */
		decRep = 0;
		for (j = 0; j < L; j++) {
			if (msk & epsilon[((i - 1)*L + j) >> auxSize]) decRep += (long)pow(2, L - 1 - j);
			msk >>= 1;
			if (msk == 0) msk = 1 << 31;
		}
		T[decRep] = i;

	}
	for (i = Q + 1; i <= Q + K; i++) { 	/* PROCESS BLOCKS */
		decRep = 0;
		for (j = 0; j < L; j++) {
			if (msk & epsilon[((i - 1)*L + j) >> auxSize]) decRep += (long)pow(2, L - 1 - j);
			msk >>= 1;
			if (msk == 0) msk = 1 << 31;
		}
		sum += log(i - T[decRep]) / log(2);
		T[decRep] = i;
	}
	phi = (double)(sum / (double)K);

	arg = fabs(phi - expected_value[L]) / (sqrt2 * sigma);
	p_value = erfc(arg);


	free(T);

	if (p_value > 0.01)
	{
		return 1;
	}
	else
	{

		return 0;
	}
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
						     S E R I A L  T E S T           CORRECT
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

double psi2(unsigned long int *epsilon, int m, int n);

int Serial(unsigned long int *epsilon, int m, int n) {
	double	p_value1, p_value2, psim0, psim1, psim2, del1, del2;

	int *auxBinary;
	unsigned long int msk = 1 << 31;
	auxBinary = (int *)malloc(n * sizeof(int));

	int k;
	for (k = 0; k < n; k++)
	{
		if (msk & epsilon[k >> auxSize])
		{
			auxBinary[k] = 1;
		}
		else
		{
			auxBinary[k] = 0;
		}
		msk >>= 1;
		if (msk == 0) msk = 1 << 31;
	}

	psim0 = psi2(auxBinary, m, n);
	psim1 = psi2(auxBinary, m - 1, n);
	psim2 = psi2(auxBinary, m - 2, n);
	del1 = psim0 - psim1;
	del2 = psim0 - 2.0*psim1 + psim2;
	p_value1 = cephes_igamc(pow(2, m - 1) / 2, del1 / 2.0);
	p_value2 = cephes_igamc(pow(2, m - 2) / 2, del2 / 2.0);

	if (p_value1 > 0.01 && p_value2 > 0.01)
	{
		return 1;
	}
	else
	{
		return 0;
	}

}

double psi2(unsigned long int *epsilon, int m, int n) {
	int				i, j, k, powLen;
	double			sum, numOfBlocks;
	unsigned int	*P;

	if ((m == 0) || (m == -1))
		return 0.0;
	numOfBlocks = n;
	powLen = (int)pow(2, m + 1) - 1;
	if ((P = (unsigned int*)calloc(powLen, sizeof(unsigned int))) == NULL) {
		return 0.0;
	}
	for (i = 1; i < powLen - 1; i++)
		P[i] = 0;	  /* INITIALIZE NODES */
	for (i = 0; i < numOfBlocks; i++) {		 /* COMPUTE FREQUENCY */
		k = 1;
		for (j = 0; j < m; j++) {
			if (epsilon[(i + j) % n] == 0)
				k *= 2;
			else if (epsilon[(i + j) % n] == 1)
				k = 2 * k + 1;
		}
		P[k - 1]++;
	}
	sum = 0.0;
	for (i = (int)pow(2, m) - 1; i < (int)pow(2, m + 1) - 1; i++)
		sum += pow(P[i], 2);
	sum = (sum * pow(2, m) / (double)n) - (double)n;
	free(P);

	return sum;
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
				A P P R O X I M A T E  E N T R O P Y   T E S T  CORRECT
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

int ApproximateEntropy(unsigned long int *epsilon, int m, int n) {
	int				i, j, k, r, blockSize, seqLength, powLen, index;
	double			sum, numOfBlocks, ApEn[2], apen, chi_squared, p_value;
	unsigned int	*P;

	seqLength = n;
	r = 0;

	int *auxBinary;
	unsigned long int msk = 1 << 31;
	auxBinary = (int *)malloc(n * sizeof(int));

	for (k = 0; k < n; k++)
	{
		if (msk & epsilon[k >> auxSize])
		{
			auxBinary[k] = 1;
		}
		else
		{
			auxBinary[k] = 0;
		}
		msk >>= 1;
		if (msk == 0) msk = 1 << 31;
	}

	for (blockSize = m; blockSize <= m + 1; blockSize++) {
		if (blockSize == 0) {
			ApEn[0] = 0.00;
			r++;
		}
		else {
			numOfBlocks = (double)seqLength;
			powLen = (int)pow(2, blockSize + 1) - 1;
			if ((P = (unsigned int*)calloc(powLen, sizeof(unsigned int))) == NULL) {
				return 0;
			}
			for (i = 1; i < powLen - 1; i++)
				P[i] = 0;
			for (i = 0; i < numOfBlocks; i++) { /* COMPUTE FREQUENCY */
				k = 1;
				for (j = 0; j < blockSize; j++) {
					k <<= 1;
					if ((int)auxBinary[(i + j) % seqLength] == 1)
						k++;
				}
				P[k - 1]++;
			}
			/* DISPLAY FREQUENCY */
			sum = 0.0;
			index = (int)pow(2, blockSize) - 1;
			for (i = 0; i < (int)pow(2, blockSize); i++) {
				if (P[index] > 0)
					sum += P[index] * log(P[index] / numOfBlocks);
				index++;
			}
			sum /= numOfBlocks;
			ApEn[r] = sum;
			r++;
			free(P);
		}
	}
	apen = ApEn[0] - ApEn[1];
	free(auxBinary);

	chi_squared = 2.0*seqLength*(log(2) - apen);
	p_value = cephes_igamc(pow(2, m - 1), chi_squared / 2.0);

	if (m > (int)(log(seqLength) / log(2) - 5)) return 0;

	if (p_value > 0.01)
	{
		return 1;
	}
	else
	{
		return 0;
	}


}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
		  N O N O V E R L A P P I N G  T E M P L A T E  T E S T
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

double NonOverlappingTemplateMatchings(unsigned long int *epsilon, int m, int n) {
	 int		numOfTemplates[100] = { 0, 0, 2, 4, 6, 12, 20, 40, 74, 148, 284, 568, 1116,
						 2232, 4424, 8848, 17622, 35244, 70340, 140680, 281076, 562152 };
	 //----------------------------------------------------------------------------
	 //NOTE:  Should additional templates lengths beyond 21 be desired, they must
	 //first be constructed, saved into files and then the corresponding
	 //number of nonperiodic templates for that file be stored in the m-th
	 //position in the numOfTemplates variable.
	 //----------------------------------------------------------------------------
	 unsigned int	bit, W_obs, nu[6], *Wj = NULL;
	 FILE			*fp = NULL;
	 double			sum, chi2, p_value, lambda, pi[6], varWj;
	 int				i, j, jj, k, match, SKIP, M, N, K = 5, p_valueCounter = 0;
	 BitSequence		*sequence = NULL;

	 int *auxBinary;
	 unsigned long int msk = 1 << 31;
	 auxBinary = (int *)malloc(n * sizeof(int));

	 for (k = 0; k < n; k++)
	 {
		 if (msk & epsilon[k >> auxSize])
		 {
			 auxBinary[k] = 1;
		 }
		 else
		 {
			 auxBinary[k] = 0;
		 }
		 msk >>= 1;
		 if (msk == 0) msk = 1 << 31;
	 }

	 N = 8;
	 M = n / N;

	 if ((Wj = (unsigned int*)calloc(N, sizeof(unsigned int))) == NULL) {
		 return 0;
	 }
	 lambda = (M - m + 1) / pow(2, m);
	 varWj = M * (1.0 / pow(2.0, m) - (2.0*m - 1.0) / pow(2.0, 2.0*m));
	 //sprintf_s(directory, "templates/template%d", m);

	 if (((isNegative(lambda)) || (isZero(lambda))) ||
		 ((fp = fopen("template9", "r")) == NULL) ||
		 ((sequence = (BitSequence *)calloc(m, sizeof(BitSequence))) == NULL)) {
		 if (sequence != NULL)
			 free(sequence);
	 }
	 else {
		 if (numOfTemplates[m] < MAXNUMOFTEMPLATES)
			 SKIP = 1;
		 else
			 SKIP = (int)(numOfTemplates[m] / MAXNUMOFTEMPLATES);
		 numOfTemplates[m] = (int)numOfTemplates[m] / SKIP;

		 sum = 0.0;
		 for (i = 0; i < 2; i++) {                      // Compute Probabilities
			 pi[i] = exp(-lambda + i * log(lambda) - cephes_lgam(i + 1));
			 sum += pi[i];
		 }
		 pi[0] = sum;
		 for (i = 2; i <= K; i++) {                      // Compute Probabilities
			 pi[i - 1] = exp(-lambda + i * log(lambda) - cephes_lgam(i + 1));
			 sum += pi[i - 1];
		 }
		 pi[K] = 1 - sum;

		 for (jj = 0; jj < MIN(MAXNUMOFTEMPLATES, numOfTemplates[m]); jj++) {
			 sum = 0;

			 for (k = 0; k < m; k++) {
				 fscanf_s(fp, "%d", &bit);
				 sequence[k] = bit;
			 }
			 for (k = 0; k <= K; k++)
				 nu[k] = 0;
			 for (i = 0; i < N; i++) {
				 W_obs = 0;
				 for (j = 0; j < M - m + 1; j++) {
					 match = 1;
					 for (k = 0; k < m; k++) {

						 if ((int)sequence[k] != (int)auxBinary[i*M + j + k]) {



							 match = 0;
							 break;
						 }
					 }
					 if (match == 1) {
						 W_obs++;
						 j += m - 1;
					 }
				 }
				 Wj[i] = W_obs;
			 }
			 sum = 0;
			 chi2 = 0.0;                                   // Compute Chi Square
			 for (i = 0; i < N; i++) {
				 chi2 += pow(((double)Wj[i] - lambda) / pow(varWj, 0.5), 2);
			 }
			 p_value = cephes_igamc(N / 2.0, chi2 / 2.0);
			 if (p_value > 0.01) p_valueCounter++;
				
			 if (SKIP > 1)
				 fseek(fp, (long)(SKIP - 1) * 2 * m, SEEK_CUR);
		 }
	 }

	 double result = (double)(p_valueCounter) / (double)(MIN(MAXNUMOFTEMPLATES, numOfTemplates[m]));

	
	 
	 
	
	 if (sequence != NULL)
		 free(sequence);

	 free(Wj);
	 if (fp != NULL)
		 fclose(fp);

	 free(auxBinary);

	 return result;

 }

 /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
				O V E R L A P P I N G  T E M P L A T E  T E S T
  * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

double	Pr(int u, double eta);

int OverlappingTemplateMatchings(unsigned long int *epsilon, int m, int n) {
	int				i, k, match;
	double			W_obs, eta, sum, chi2, p_value, lambda;
	int				M, N, j, K = 5;
	unsigned int	nu[6] = { 0, 0, 0, 0, 0, 0 };
	double			pi[6] = { 0.364091, 0.185659, 0.139381, 0.100571, 0.0704323, 0.139865 };
	BitSequence		*sequence;

	int *auxBinary;
	unsigned long int msk = 1 << 31;
	auxBinary = (int *)malloc(n * sizeof(int));

	for (k = 0; k < n; k++)
	{
		if (msk & epsilon[k >> auxSize])
		{
			auxBinary[k] = 1;
		}
		else
		{
			auxBinary[k] = 0;
		}
		msk >>= 1;
		if (msk == 0) msk = 1 << 31;
	}


	M = 1032;
	N = n / M;

	if ((sequence = (BitSequence *)calloc(m, sizeof(BitSequence))) == NULL) {
		return 0;
	}
	else
		for (i = 0; i < m; i++)
			sequence[i] = 1;

	lambda = (double)(M - m + 1) / pow(2, m);
	eta = lambda / 2.0;
	sum = 0.0;
	for (i = 0; i < K; i++) {			// Compute Probabilities 
		pi[i] = Pr(i, eta);
		sum += pi[i];
	}
	pi[K] = 1 - sum;

	for (i = 0; i < N; i++) {
		W_obs = 0;
		for (j = 0; j < M - m + 1; j++) {
			match = 1;
			for (k = 0; k < m; k++) {
				if (sequence[k] != auxBinary[i*M + j + k])
					match = 0;
			}
			if (match == 1)
				W_obs++;
		}
		if (W_obs <= 4)
			nu[(int)W_obs]++;
		else
			nu[K]++;
	}
	sum = 0;
	chi2 = 0.0;                                   // Compute Chi Square 
	for (i = 0; i < K + 1; i++) {
		chi2 += pow((double)nu[i] - (double)N*pi[i], 2) / ((double)N*pi[i]);
		sum += nu[i];
	}
	p_value = cephes_igamc(K / 2.0, chi2 / 2.0);


	free(sequence);
	free(auxBinary);

	if (p_value > 0.01)
	{
		return 1;
	}
	else
	{

		return 0;
	}

}

double Pr(int u, double eta) {
	int		l;
	double	sum, p;

	if (u == 0)
		p = exp(-eta);
	else {
		sum = 0.0;
		for (l = 1; l <= u; l++)
			sum += exp(-eta - u * log(2) + l * log(eta) - cephes_lgam(l + 1) + cephes_lgam(u) - cephes_lgam(l) - cephes_lgam(u - l + 1));
		p = sum;
	}
	return p;
}

