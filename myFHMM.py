
# Partial Mean Field Factorial Hidden Markov Model 
# X - N x p data matrix
# T - length of each sequence 
# N - Number of sequences (N must evenly divide by T, default T=N)
# M - number of chains (default 2)
# K - number of states per chain (default 2)
# cyc - maximum number of cycles of Baum-Welch (default 100)
# tol - termination tolerance (prop change in likelihood) (default 0.0001)
# iter - maximum number of MF iterations (default 10)
#
# Mu - mean vectors
# Cov - output covariance matrix (full, tied across states)
# P - state transition matrix
# Pi - priors
# LL - log likelihood curve
# TL - true log likelihood curve (optional -- slows computation)

# Iterates until a proportional change < tol in the log likelihood 
# or cyc steps of Baum-Welch

# This version has automatic MF stopping 
import numpy as np
from numpy.linalg import inv
from numpy.linalg import det
X=np.loadtxt('mydata.txt')
M=3
K=2
T=0
def PMFHMM(X,T=0,M=2,K=2,cycleBW=100,tol=0.0001,iter=10):
  N,D=X.shape
  if T==0:
    T=N
  
  N=N/T
  tiny=np.exp(-700)
  Cov=np.diag(np.diag(np.cov(X.T)))

  XX=X.T.dot(X)/(N*T)

  Mu=np.dot(np.random.normal(size=(M*K,D)),np.sqrt(Cov)/M)+np.ones((K*M,1))*np.mean(X,axis=0)/M  # Mu (M*K,D)

  Pi=np.random.random((K,M))# priors of each m markov process
  Pi=Pi/Pi.sum(axis=0)

  P=np.random.random((K*M,K))# transition matrix 
  P=(P.T/P.sum(axis=1)).T

  
  lik=0
  logLik=[]
  trueLoglik=[]

  gamma=np.zeros((N*T,K*M))    # P ( s_i | theta) Gamma is (T*N,K*M)
  k1=(2*np.pi)**(-D/2) # constant in equation 4a

  mf=np.ones((N*T,M*K))/K # initialize mean field Q({S_t} | theta)
  h=np.ones((N*T,M*K))/K # initialize probability of being in each state (K) for each variable (M) at each time step (N) 
  
  alpha=np.zeros((N*T,M*K)) # initialize alpha and beta for the Baum_welch algorithm
  beta=np.zeros((N*T,M*K))
  logMF=np.log(mf)
  expH=np.exp(h)

  for cycle in xrange(cycleBW):
  
   # FORWARD-BACKWARD %%% MF E STEP 
    gammaNew=[]
    GammaX=np.zeros((M*K,D))
    Eta=np.zeros((K*M,K*M))

    # Solve mean field equations for all input sequences
    inverseCov=inv(Cov)  #C^-1  
    k2=k1/np.sqrt(det(Cov)) # constant 
    eN=np.ones((1,N))
    iterMF=iter
    
    for l in xrange(iter):
      
      mf0=mf # initiate mean field
      logMF0=logMF # initiate log mean field

      # first compute h values based on mf

      for i in range(T):
        d2=range(i*N,(i+1)*N)
        
        X_i = X[d2,:] # N*D
        yHat=np.dot(mf[d2,:],Mu) # compute yHat (N*D)
          
        for j in range(M):
          
          d1=range(j*K,(j+1)*K)
          
          Mu_j=Mu[d1,:] # K*D
          mfOld=mf[d2,d1]
          
          logP=np.log(P[d1,:]+tiny)
          logPi=np.log(Pi[:,j]+tiny)
          
          Delta=(np.diag(np.dot(np.dot(Mu_j,inverseCov),Mu_j.T))*eN)
          h[d2,d1]=np.dot(np.dot(Mu_j,inverseCov),(X_i-yHat).T).T+np.dot(np.dot(np.dot(Mu_j,inverseCov),Mu_j.T),mf[d2,d1].T)- 0.5*Delta # equation 12a
          h[d2,d1]-=np.max(h[d2,d1])
          
      expH=np.exp(h)

        # then compute mf values based on h using Baum-Welch
    
      scale=np.zeros((T*N,M))

      for j in xrange(M):
        
        d1=range(j*K,(j+1)*K)
        
        for i in xrange(T):
          
          d2=range(i*N,(i+1)*N)
          if i==0:
            alpha[d2,d1]=expH[d2,d1]*(np.ones((N,1))*Pi[:,j].T)
          else:
            alpha[d2,d1]=np.dot(alpha[[g-N for g in d2],d1],P[d1,:])*expH[d2,d1]
          
          if N!=1:
            scale[d2,j]=alpha[d2,d1].sum(axis=1)+tiny
          else:
            scale[d2,j]=alpha[d2,d1].sum()+tiny

          alpha[d2,d1]=alpha[d2,d1]/scale[d2,j]

        for i in reversed(xrange(T)):
          
          d2=range(i*N,(i+1)*N)
          
          if i==np.max(xrange(T)):
            beta[d2,d1]=np.ones((N,K))/scale[d2,j]
          else:
            beta[d2,d1]=np.dot(beta[[g+N for g in d2],d1]*expH[[g+N for g in d2],d1],P[d1,:].T)
            beta[d2,d1]=beta[d2,d1]/scale[d2,j]
    
        mf[:,d1]=alpha[:,d1]*beta[:,d1]
        mf[:,d1]=(mf[:,d1].T/(mf[:,d1].sum(axis=1)+tiny)).T  
      
      mask0=(mf==0)*1
      logMF=np.log(mf+mask0*tiny)
      delMF=(mf*logMF).sum(axis=0).sum()-(mf*logMF0).sum(axis=0).sum()

      if delMF<N*T*1e-6
        iterMF=l
        break

  # calculating mean field log likelihood 
    
    oldLik=lik
    lik=calcmflike(Xalt,T,mf,M,K,Mu,Cov,P,Pi); # to modify!
  
  #  first and second order correlations - P (s_i, s_j | O)

    gamma=mf
    Gamma=gamma
    Eta=np.dot(gamma.T,gamma)
    gammaSum=gamma.sum(axis=0)

  for j in xrange(M):
    d2=range(j*K,(j+1)*K)
    Eta[np.ix_(d2,d2)]=np.diag(gammaSum[d2]) 

  
  GammaX=np.dot(gamma.T,X)
  Eta=(Eta+Eta.T)/2

  X_j=np.zeros((M*K,K))
  for i in range(T-1):
    d1=range(i*N,(i+1)*N)
    d2=range((i+1)*N,(i+2)*N)
    for j in range(M):
      jK=range(j*K,(j+1)*K)
      # t=gamma(d1,jK)'*gamma(d2,jK);
      t = P[jK,:]*np.dot(alpha[d1,jK].T,(beta[d2,jK]*expH[d2,jK]))
      X_j[jK,:]=X_j[jK,:]+t/np.sum(t)


  
  # Calculate Likelihood and determine convergence
  
  logLik.append(lik)
  if (nargout>=6)
    truelik=calclike(X,T,M,K,Mu,Cov,P,Pi);
    TL=[TL truelik];
    fprintf('cycle %i mf iters %i log like= %f true log like= %f',cycle,itermf,lik,truelik);  
  elseif (nargout==5)
    fprintf('cycle %i mf iters %i log likelihood = %f ',cycle,itermf,lik);
  else
    fprintf('cycle %i mf iters %i ',cycle,itermf);
  end;
  
  if (nargout>=5)
    if (cycle<=2)
      likbase=lik;
    elseif (lik<oldlik-2)
      fprintf('v');
    elseif (lik<oldlik) 
      fprintf('v');
    elseif ((lik-likbase)<(1 + tol)*(oldlik-likbase)) 
      fprintf('\n');
      break;
    end;
  end;
  fprintf('\n');