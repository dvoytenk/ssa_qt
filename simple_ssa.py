from numpy import *
from matplotlib.pyplot import *

#simple SSA example, use gui version to play with different values
#NOTE THAT DATA SHOULD HAVE THE SAME INTERVAL BETWEEN MEASUREMENTS (e.g, 1 minute), IF NOT, MUST RESAMPLE IT TO HAVE SUCH

filename='example.txt'

#input min/max eigenvalues to use (top eigenvalues are the strongest parts of the signal)
nmin=0
nmax=4

orig_ts=loadtxt(filename)
ts_len=len(orig_ts)

#this assumes a window of half of the time series length
winsize=(size(orig_ts)/2)
win_len=int(winsize)

#this builds a specially banded Hankel matrix (slow, other methods available)
nrows=ts_len-win_len+1
ncols=win_len
hankel_mat=zeros([nrows,win_len])
for i in range(0,nrows):
    for j in range(0,ncols):
        hankel_mat[i,j]=i+j


z=hankel_mat
z=z.transpose()
z=z.astype('int')

#this bands the data (slow, faster methods available) to match the indexing of the Hankel matrix
#this basically wraps the data into a matrix
matdim=shape(z)
nrows=matdim[0]
ncols=matdim[1]
tsmat=zeros(shape(z))
for i in range(0,nrows):
  tsmat[i,:]=orig_ts[list(z[i,:].astype('int'))]

V=dot(tsmat,transpose(tsmat))
#do SVD on the formatted data matrix
U,s,V=np.linalg.svd(V)


#get matrix of eigenvalues to use
S=zeros([nrows,ncols])
for i in range(nmin,nmax):
    S[i,i]=s[i]
    
#reconstruct data set
R=np.dot(U, np.dot(S, V))
svd_ts=zeros(shape(orig_ts))
for i in range(0,nrows):
    for j in range(0,ncols):
        svd_ts[z[i,j]]=R[i,j]
    
plot(orig_ts,'*')    
plot(svd_ts)
legend(['original','filtered'])
show()

#to save
savetxt('filtered_'+str(nmin)+'_'+str(nmax)+'_'+filename, svd_ts)
