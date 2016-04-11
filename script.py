import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
import math
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys
#matplotlib inline

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD
    maxi=int(np.amax(y));
    T=X.shape[1]
    means=np.empty((maxi,T));
    y=y.flatten();
    for i in range(maxi):
        meant=X[y==i+1];
        means[i,:]=np.mean(meant,axis=0);
    covmat=np.cov(X,rowvar=0);
    means=means.transpose() 
    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    maxi=int(np.amax(y));
    T=X.shape[1]
    means=np.empty((maxi,T));
    covmats=[np.zeros((X.shape[1],X.shape[1]))] * maxi;
    #print covmats
    y=y.flatten();
    for i in range(maxi):
        meant=X[y==i+1];
        means[i,:]=np.mean(meant,axis=0);
        covmats[i]=np.cov(meant,rowvar=0);
    means=means.transpose()    
    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels
    
    # IMPLEMENT THIS METHOD
    N = Xtest.shape[0];
    ypred=np.zeros((N,1));
    classes = means.shape[1];
    Correct = 0.0;
    inverse = inv(covmat);
    ytest = ytest.astype(int);
    for i in range (0, N ):
        pdf = 0;
        classnumber = 0;
        for j in range (1, classes+1):
            result = np.exp((-1/2)*np.dot(np.dot(np.transpose(Xtest[i,:].transpose() - means[:, j-1]),inverse),(Xtest[i,:].transpose() - means[:, j-1])));
            if (result > pdf):
                classnumber = j;
                pdf = result;
                ypred[i]=j           
        if (classnumber == ytest[i]):
            Correct = Correct + 1;
            ypred[i]=classnumber;
         
    acc = Correct/N*100;
    ypred=ypred.flatten();
    ypred = ypred.astype(int);
    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    classes = np.shape(means)[1];
    denominator = np.zeros(classes);
    
    for i in range (classes):
        d = np.shape(covmats[i])[0];
        denominator[i] = 1.0/(np.power(2*np.pi, d/2)*np.power(det(covmats[i]),1/2));

        covmats[i] = inv(covmats[i]);
    N = Xtest.shape[0];
    ypred=np.zeros((N,1));
    
    Correct = 0.0;
    ytest = ytest.astype(int);
    for i in range ( N ):
        pdf = 0;
        classnumber = 0;
        for k in range (1, classes+1):
            inverse = covmats[k-1];
            numerator=np.exp((-1/2)*np.dot(np.dot(np.transpose(Xtest[i,:].transpose() - means[:, k-1]),inverse),(Xtest[i,:].transpose() - means[:, k-1])));
            result = denominator[k-1]*numerator;
            if (result > pdf):
                classnumber = k;
                pdf = result;
                ypred[i]=k;
        if (classnumber == ytest[i]):
            Correct = Correct + 1;  
           
    acc = Correct/N*100;
    ypred=ypred.flatten();
    ypred = ypred.astype(int);
    
    return acc, ypred

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1                                                                
    # IMPLEMENT THIS METHOD 
    Xmult=np.dot(X.transpose(),X);
    inverse=inv(Xmult);
    w=np.dot(inverse,np.dot(X.transpose(),y));                                                  
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                             

    # IMPLEMENT THIS METHOD
    multXtX=np.dot(X.transpose(),X)
    I=np.identity(multXtX.shape[0]);
    inverse=inv(lambd*I+multXtX);
    #print y.shape[:]
    w=np.dot(inverse,np.dot(X.transpose(),y));                                                     
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # rmse
    # IMPLEMENT THIS METHOD
    N=Xtest.shape[0]
    multwx=np.dot(Xtest,w);
    sq=np.square(ytest-multwx);
    rmse=math.sqrt(np.sum(sq)/N);
    return rmse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD  
    #print w.shape[:]
    #w=np.mat(w)
    y=y.flatten()
    w=w.T
    multxw=np.dot(X,w);
    subtract_ymu=y-multxw;
    error=np.dot(subtract_ymu.T,subtract_ymu)/(2)+lambd*np.dot(w.T,w)/2;
    error_grad=np.dot(w.T,np.dot(X.T,X))-np.dot(y.T,X)+lambd*w.T;     
    error_grad=np.array(error_grad).flatten();
    error=np.array(error).flatten(); 
    return error, error_grad
    
def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (d+1))                                                         
    # IMPLEMENT THIS METHOD
    Xd=np.zeros(shape=(x.shape[0],p+1))
    for i in range(x.shape[0]):
        for j in range(p+1):
            Xd[i][j]=np.power(x[i],j)
    return Xd

# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
#zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])))
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)

plt.show()

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])))
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.show()

# Problem 2

if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle_train=testOLERegression(w,X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i_train=testOLERegression(w_i,X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)
print('RMSE with intercept for Training  '+str(mle_i_train))
print('RMSE without intercept for Training  '+str(mle_train))
print('RMSE without intercept for Testing  '+str(mle))
print('RMSE with intercept for Testing '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
w_l = learnRidgeRegression(X_i,y,0.06)
mle_i_rid=testOLERegression(w_l,Xtest_i,ytest)
mle_i_rid_tr=testOLERegression(w_l,X_i,y)
print('RMSE Test with intercept for Ridge Regression '+str(mle_i_rid))
print('RMSE Train with intercept for Ridge Regression '+str(mle_i_rid_tr))
rmses3 = np.zeros((k,1))
lambda3=np.zeros((k,1));
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    
    rmses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    # print rmses3[i], lambda3[i] 
    i = i + 1

plt.plot(lambdas,rmses3)
plt.show()


# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
rmses4 = np.zeros((k,1))
rmses4t=np.zeros((k,1));
opts = {'maxiter' : 100}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    rmses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
plt.plot(lambdas,rmses4)

plt.show()


# Problem 5
pmax = 7
lambda_opt = lambdas[np.argmin(rmses4)]
rmses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    rmses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    rmses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
    #print rmses5[p,0],rmses5[p,1]
plt.plot(range(pmax),rmses5)
plt.legend(('No Regularization','Regularization'))
plt.show()
