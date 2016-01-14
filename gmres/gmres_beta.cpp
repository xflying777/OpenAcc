#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
int show1(double *b , int N);
int show(double **A,int N1, int N2);
int identity(double **A,int N);
double norm(double *a, int N);
int backsubtitution(double **R, double *b, double *x, int N);
int multiplyvector(double **A,double *b, double *b1,int N1, int N2);
int kron(double **A, double **B, double **C, int N);
int empty(double **A, int N1, int N2);
double inner(double *a,double *b,int N);
int transpose(double **A,int N);
int multiply(double **A, double **B, double **C, int N1, int N2, int N3);

int arnoldi(double **A,double **H, double **Q, double *b, int M, int N);
int houseQR(double **A, double **Q ,double **R, int N1, int N2);
int solvebyQR(double **A, double *b,double *x, int N1, int N2);
int GMRES(double **A, double *b, double *x, int M, int N ,double tol);

void bfunc(double *x, int N);
void ufunc(double *u, int N);
double error(double *x , double *u, int N);


int main(int argc, char *argv[]) 
{

	int i,j,k,N;
	double h, *b, *x, *u;
	static double **A;
	
	
	printf("請輸入要分割的數量\n");
	scanf("%d",&N);
	
	b=(double*)malloc(N*sizeof(double));
	u=(double*)malloc(N*sizeof(double));
	x=(double*)malloc(N*sizeof(double));
	
	A=(double**)malloc(N*sizeof(double*));
	for(i=0;i<N;i++)
	A[i]=(double*)malloc(N*sizeof(double));
	

	//開始製造拿來離散的A 
	for(i=0;i<N;i++)
	{
		for(j=0;j<N;j++)
		{
			A[i][j]=0;
		}
	}
	for(i=0;i<N;i++)	
	A[i][i]=-2;
	
	for(i=0;i<N-1;i++)
	A[i][i+1]=1;
	
	for(i=0;i<N-1;i++)
	A[i+1][i]=1;
	
	h=M_PI/(N+1);
	for(i=0;i<N;i++)
	{
		for(j=0;j<N;j++)
		{
			A[i][j]=A[i][j]/(h*h);
		}
	}
	
	//結束製造拿來離散的A
	

	bfunc( b, N);
	ufunc( u, N);
	
	GMRES(A, b, x, N, N-1 , 1.0e-6);

	printf(" error = %e \n", error(x, u, N));
	

	

	
	system("PAUSE");
	return 0;
}


//自己能夠運作的函數 開始
double error(double *x , double *u, int N)
{
	int i;
	double error, temp;
	error=0.0;
	
	for(i=0;i<N;i++)
	{
		temp=fabs(x[i]-u[i]);
		if(temp > error) error = temp;
	}
	return error;
}

void bfunc(double *x,int N)
{
	int i;
	double h, temp;
	h=M_PI/(N+1);
	
	for(i=0;i<N;i++)
	{
		temp = (i+1)*h;
		x[i] = 2*cos(temp)-temp*sin(temp);
	}
}

void ufunc(double *u, int N)
{
	int i;
	double h, temp;
	h=M_PI/(N+1);
	for(i=0;i<N;i++)
	{
		temp = (i+1)*h;
		u[i] = temp*sin(temp);	
	}
}

int show1(double *b , int N)
{
	int i;
	for(i=0;i<N;i++)
	{
		printf(" %f ",b[i]);
	}
	printf("\n");
	return 0;
}
int show(double **A,int N1, int N2)
{
	int i, j;
	for(i=0;i<N1;i++)
	{
		for(j=0;j<N2;j++)
		{
			printf(" %f ",A[i][j]);
		}
		printf("\n");
	}
	return 0;
}
int identity(double **A,int N)
{
	int i,j;
	for(i=0;i<N;i++)
	{
		for(j=0;j<N;j++)
		{
			if(i==j)
			{
				A[i][j]=1;
			}
			else
			{
				A[i][j]=0;
			}
		}
	}
	return 0;
}
double norm(double *a, int N)
{
	int i;
	double total=0;
	for(i=0;i<N;i++)
	{
		total=total+a[i]*a[i];
	}
	total=sqrt(total);
	
	return total;
}
int backsubtitution(double **R, double *b, double *x, int N)
{
	int i, j, I;
	
	for(i=0;i<N;i++)
	{
		I=N-i-1;
		x[I]=b[I];
		for(j=I+1;j<N+1;j++)
		{
			x[I]=x[I]-x[j]*R[I][j];	
		}
		
		x[I]=x[I]/R[I][I];

	}
	return 0;
}
int multiplyvector(double **A,double *b, double *b1,int N1, int N2)
{
	int i,j;
	double t;
	for(i=0;i<N1;i++)
	{
		t=0;
		for(j=0;j<N2;j++)
		{
			t=t+A[i][j]*b[j];
		}
		b1[i]=t;
	}
	
	return 0;
}

int kron(double **A, double **B, double **C, int N)
{
	int i,j,k,l;
	static double**T1;
	T1=(double**)malloc(N*sizeof(double*));
	for(i=0;i<N;i++)
	T1[i]=(double*)malloc(N*sizeof(double));
	
	for(i=0;i<N;i++)
	{
		for(j=0;j<N;j++)
		{
			for(k=0;k<N;k++)
			{
				for(l=0;l<N;l++)
				{
					T1[k][l]=A[i][j]*B[k][l];
				}
			}
			//把T1填入 C中對應的位置 
			for(k=0;k<N;k++)
			{
				for(l=0;l<N;l++)
				{
					C[i*N+k][j*N+l]=T1[k][l];
				}
			}		
		}
	}
	return 0;
}
int empty(double **A, int N1, int N2)
{
	int i,j;
	for(i=0;i<N1;i++)
	{
		for(j=0;j<N2;j++)
		{
			A[i][j]=0;
		}
		
	}
	return 0;
}
double inner(double *a,double *b,int N)
{
	int i;
	double ans=0;
	for(i=0;i<N;i++)
	{
		ans=ans+a[i]*b[i];
	}
	return ans;
}

int transpose(double **A,int N)
{
	int i,j;
	static double **B;
	B = (double**)malloc(N*sizeof(double*));
	for(i=0;i<N;i++)
	{
		B[i] = (double*)malloc(N*sizeof(double));
	}
	for(i=0;i<N;i++)
	{
		for(j=0;j<N;j++)
		{
			B[i][j]=A[j][i];
		}
	}
	
	for(i=0;i<N;i++)
	{
		for(j=0;j<N;j++)
		{
			A[i][j]=B[i][j];
		}
	}
	
	
	return 0;
}

int multiply(double **A, double **B, double **C, int N1, int N2, int N3)
{
	int i, j, k;
	double S1;
	for(i=0;i<N1;i++)
	{
		for(j=0;j<N3;j++)
		{
			S1=0;
			for(k=0;k<N2;k++)
			{
				S1=S1+A[i][k]*B[k][j];
			}
			C[i][j]=S1;
		}
	}
	return 0;
}

//自己能夠運作的函數 結束 


int arnoldi(double **A,double **H, double **Q, double *b, int M, int N)
{
	int i,j,k;
	double t, *b1, *b2, *b3;
	static double **T1, **T2, **T3, **T4;
	
	b1=(double*)malloc(M*sizeof(double));
	b2=(double*)malloc(M*sizeof(double));
	b3=(double*)malloc(M*sizeof(double));
	
	
	empty(H,N+1,N);
	
	
	t=norm(b,M);
	for(i=0;i<M;i++)
	{
		Q[i][0]=b[i]/t;
	}
	
	
	for(i=0;i<N;i++)
	{
		for(j=0;j<M;j++)
		{
			b1[j]=Q[j][i];
		}
		
		multiplyvector(A,b1,b2,M,M);
		
		for(j=0;j<i+1;j++)
		{
			for(k=0;k<M;k++)
			{
				b1[k]=Q[k][j];
			}
			H[j][i]=inner(b2,b1,M);

			for(k=0;k<M;k++)
			{
				b2[k]=b2[k]-H[j][i]*b1[k];
			}
			
		}
		H[i+1][i]=norm(b2,M);
		for(j=0;j<M;j++)
		{
			Q[j][i+1]=b2[j]/H[i+1][i];
		}

	}
	
	return 0;
}


int houseQR(double **A, double **Q ,double **R, int N1, int N2)
{
	int i, j, k, J; 
	double S1;
	double *a, *a1, *a2;
	a  = (double*)malloc(N1*sizeof(double));
	a1 = (double*)malloc(N1*sizeof(double));
	a2 = (double*)malloc(N1*sizeof(double));
	
	static double **T1, **T2, **Q1;
	T1 = (double**)malloc(N1*sizeof(double*));
	T2 = (double**)malloc(N1*sizeof(double*));
	Q1 = (double**)malloc(N1*sizeof(double*));
	for(i=0;i<N1;i++)
	{
		T1[i] = (double*)malloc(N1*sizeof(double));
		T2[i] = (double*)malloc(N1*sizeof(double));
		Q1[i] = (double*)malloc(N1*sizeof(double));
	}
	
	identity(Q1,N1);
	for(i=0;i<N1;i++)
	{
		for(j=0;j<N2;j++)
		{
			R[i][j]=A[i][j];
		}
	}
	
	for(i=0;i<N2;i++)
	{
		J=N1-i; //J為 要拿來當作 旋轉鏡射矩陣的  向量維度 
		for(j=0;j<J;j++)
		{
			a[j]=R[j+i][i];
		}
		S1=0;
		for(j=0;j<J;j++)
		{
			S1=S1+a[j]*a[j];
		}
		S1=sqrt(S1);
		
		a1[0]=S1;
		for(j=1;j<J;j++)
		{
			a1[j]=0;
		}
	
		if(a[0]>=0) a1[0]=-a1[0];
	
		for(j=0;j<J;j++)
		{
			a2[j]=a[j]-a1[j];
		}
		S1=0;
		for(j=0;j<J;j++)
		{
			S1=S1+a2[j]*a2[j];
		}
		for(j=0;j<J;j++)
		{
			for(k=0;k<J;k++)
			{
				T1[j][k]=a2[j]*a2[k]/S1;
			}
		}
		//T2 先做成單位矩陣
		identity(T2, N1);
		for(j=0;j<J;j++)
		{
			for(k=0;k<J;k++)
			{
				T2[j+i][k+i]=T2[j+i][k+i]-2*T1[j][k];
			}
		}
		multiply(T2, R, T1, N1, N1, N2);
		for(j=0;j<N1;j++)
		{
			for(k=0;k<N2;k++)
			{
				R[j][k]=T1[j][k];
			}
		}
		transpose(T2,N1);
		multiply(Q1, T2, T1, N1, N1, N1);
		for(j=0;j<N1;j++)
		{
			for(k=0;k<N1;k++)
			{
				Q1[j][k]=T1[j][k];
			}
		}	
	}
	
	for(i=0;i<N1;i++)
	{
		for(j=0;j<N1;j++)
		{
			Q[i][j]=Q1[i][j];
		}
	}
	
	return 0;		
}
int solvebyQR(double **A, double *b,double *x, int N1, int N2)
{
	int i, j, k;
	double *b1, t;
	b1 = (double*)malloc(N1*sizeof(double));
	static double **Q, **R;
	Q=(double**)malloc(N1*sizeof(double*));
	for(i=0;i<N1;i++)
	{
		Q[i]=(double*)malloc(N1*sizeof(double));
	}
	R=(double**)malloc(N1*sizeof(double*));
	for(i=0;i<N1;i++)
	{
		R[i]=(double*)malloc(N2*sizeof(double));
		
	}
	houseQR(A, Q ,R, N1,N2);
	for(i=0;i<N2;i++)
	{
		b1[i]=0;
		for(j=0;j<N1;j++)
		{
			b1[i]=b1[i]+Q[j][i]*b[j];
		}
	}
	backsubtitution(R,b1,x,N2);
	
	

	return 0;
}

// GMRES 開始
 
int GMRES(double **A, double *b, double *x, int M, int N ,double tol)
{
	int i, j, k, N1;
	double t, *bb, *x1;
	bb=(double*)malloc(M*sizeof(double));
	
	static double **H, **Q;
	Q=(double**)malloc(M*sizeof(double*));
	for(i=0;i<M;i++)
	Q[i]=(double*)malloc(M*sizeof(double));
	H=(double**)malloc((N+1)*sizeof(double*));
	for(i=0;i<N+1;i++)
	H[i]=(double*)malloc(N*sizeof(double));
	
	t=norm(b,M);
	bb[0]=t;
	
	
	for(i=1;i<N+1;i++)
	{
		arnoldi(A, H, Q, b, M, i);
		
		if(H[i][i-1]<=tol)
		{
			N1=i;
			break;
		}
		if(i==N)
		{
			N1=i;
		}
	}
	printf("疊代到第 %d 次  GMRES 收斂 \n",N1);
	
	for(i=1;i<N1;i++)
	bb[i]=0;
	
	
	x1=(double*)malloc(N1*sizeof(double));
	
	solvebyQR(H, bb, x1, N1+1, N1);
	
	multiplyvector(Q, x1, x, M ,N1);
	
	
	
	return 0;
}
// GMRES 結束 
