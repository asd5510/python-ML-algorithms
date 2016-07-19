from numpy import *
import matplotlib.pyplot as plt

class SoftmaxRegression:
    def __init__(self):
        self.dataMat = []
        self.labelMat = []
        self.weights = []
        
        self.trainMat = []
        self.trainLabelMat = []
        self.testMat = []
        self.testLabelMat = []
        
        self.DataNum = 0 #total data number 450
        self.M = 0 #train data number,405
        self.N = 0 #feature number,25
        self.K = 0 #label number,30
        self.T = 0 #test data number,45
        self.alpha = 0.001

    def loadDataSet(self,trainfile,testfile):
        for line in open(trainfile,'r'):
            items = line.strip().split(',')
            self.dataMat.append(map(float,items[0:25]))
            self.labelMat.append(int(items[-1])-1)
        for line in open(testfile,'r'):
            items = line.strip().split(',')
            self.dataMat.append(map(float,items[0:25]))
            self.labelMat.append(int(items[-1])-1)
        self.DataNum = shape(self.dataMat)[0]
    
    def shuffleDataSet(self):
        self.trainMat = []
        self.trainLabelMat = []
        self.testMat = []
        self.testLabelMat = []
        perm = arange(self.DataNum)
        random.shuffle(perm)
        for j in xrange(self.DataNum):
            if j < self.DataNum*0.9:
                self.trainMat.append(self.dataMat[perm[j]])
                self.trainLabelMat.append(self.labelMat[perm[j]])
            else:
                self.testMat.append(self.dataMat[perm[j]])
                self.testLabelMat.append(self.labelMat[perm[j]])
        self.K = len(set(self.trainLabelMat))
        self.testMat = mat(self.testMat)
        self.testLabelMat = mat(self.testLabelMat)
        self.trainMat = mat(self.trainMat)
        self.trainLabelMat = mat(self.trainLabelMat).transpose()     
        self.M,self.N = shape(self.trainMat)
        self.T = shape(self.testMat)[0]
        self.weights = mat(ones((self.N,self.K)))

        # self.weights = [[-1.19792777,6.05913226,-4.44164147,3.58043698],
 # [ 1.78758743,0.47379819,0.63335518,1.1052592 ],
 # [ 1.48741185,-0.18748907,1.79339685,0.90668037]]

    def likelihoodfunc(self):
        likelihood = 0.0
        for i in range(self.M):
            t = exp(self.dataMat[i]*self.weights)
            likelihood += log(t[0,self.labelMat[i,0]]/sum(t))
        print likelihood

    def gradientAscent(self):
        for l in range(10):
            error = exp(self.dataMat*self.weights)
            rowsum = -error.sum(axis=1)
            rowsum = rowsum.repeat(self.K, axis=1)
            error = error/rowsum
            for m in range(self.M):
                error[m,self.labelMat[m,0]] += 1
            self.weights = self.weights + self.alpha * self.dataMat.transpose()* error

            self.likelihoodfunc()
        print self.weights

    def stochasticGradientAscent_V0(self,numIter = 500):
        for l in range(numIter):
            for i in range(self.M):
                error = exp(self.trainMat[i]*self.weights)
                rowsum = -error.sum(axis=1)
                rowsum = rowsum.repeat(self.K, axis=1)
                error = error/rowsum
                error[0,self.trainLabelMat[i,0]] += 1
                self.weights = self.weights + self.alpha * self.trainMat[i].transpose()* error
                # self.likelihoodfunc()
#         print self.weights

    def stochasticGradientAscent_V1(self,numIter = 500):
        for l in range(numIter):
            idxs = range(self.M)
            for i in range(self.M):
                alpha = 4.0/(1.0+l+i)+0.01
                rdmidx = int(random.uniform(0,len(idxs)))
                error = exp(self.dataMat[rdmidx]*self.weights)
                rowsum = -error.sum(axis=1)
                rowsum = rowsum.repeat(self.K, axis=1)
                error = error/rowsum
                error[0,self.labelMat[rdmidx,0]] += 1
                self.weights = self.weights + alpha * self.dataMat[rdmidx].transpose()* error
                del(idxs[rdmidx])

                # self.likelihoodfunc()
        print self.weights

    def classify(self,X):
        p = X * self.weights
        return p.argmax(1)[0,0]

    def Mytest(self):
        acc = 0
        for i in arange(self.T):
            if self.testLabelMat[0,i] == self.classify(self.testMat[i]) :
                acc = acc + 1
#                 print self.classify(self.testMat[i]) 
        print acc/float(self.T)
        return acc/float(self.T)
        
myclassification = SoftmaxRegression()
myclassification.loadDataSet(trainfile,testfile)
acc  = 0
for i in range(20):   
    myclassification.shuffleDataSet()
    myclassification.stochasticGradientAscent_V0(1000)
    acc = acc + myclassification.Mytest()
print acc/20
