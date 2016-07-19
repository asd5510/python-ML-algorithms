import sys, random, math
from operator import itemgetter
DIMENSION = 64
SQRT_DIMENSION = math.sqrt(DIMENSION)

class LFM():
    ''' TopN recommendation - LFM '''
    def __init__(self):
        self.trainset = {}
        self.testset = {}

        self.n_sim_movie = 20
        self.n_rec_movie = 10

        self.u_f = {}
        self.i_f = {}
        self.movie_count = 0
		self.movie_popular = {}
    @staticmethod
    def loadfile(filename):
        ''' load a file, return a generator. '''
        fp = open(filename, 'r')
        for i, line in enumerate(fp):
            yield line.strip('\r\n')
            if i % 100000 == 0:
                print >> sys.stderr, 'loading %s(%s)' % (filename, i)
        fp.close()
        print >> sys.stderr, 'load %s succ' % filename


    def generate_dataset(self, filename, pivot=0.875):
        ''' load rating data and split it to training set and test set '''
        trainset_len = 0
        testset_len = 0

        for line in self.loadfile(filename):
            user, movie, rating, _ = line.split('::')
            # split the data by pivot
            if (random.random() < pivot):
                self.trainset.setdefault(user, {})
                self.trainset[user][movie] = int(rating)
                trainset_len += 1
            else:
                self.testset.setdefault(user, {})
                self.testset[user][movie] = int(rating)
                testset_len += 1

        print >> sys.stderr, 'split training set and test set succ'
        print >> sys.stderr, 'train set = %s' % trainset_len
        print >> sys.stderr, 'test set = %s' % testset_len


    def predict(self,user,item):
        self.u_f.setdefault(user, self.random_qp())#哦，这里是i不存在的时候，才会重新random赋值
        self.i_f.setdefault(item, self.random_qp())
        return sum(self.u_f[user][f]*self.i_f[item][f] for f in xrange(DIMENSION))

    def recommend(self,user):
    	rank = {}
        K = self.n_sim_movie
        N = self.n_rec_movie
        for i in self.i_f:
        	product = map(lambda (a,b):a*b,zip(self.u_f[user],self.i_f[i]))
        	rank[i] = sum(product)
        # for f,puf in self.u_f[user].items():
        # 	for i,qfi in self.i_f[f].items():
        # 		if i not in rank:
        # 			rank[i] += puf*qfi
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[:N]
    def evaluate(self):
	    ''' return precision, recall, coverage and popularity '''
	    print >> sys.stderr, 'Evaluation start...'

	    N = self.n_rec_movie
	    #  varables for precision and recall 
	    hit = 0
	    rec_count = 0
	    test_count = 0
	    # varables for coverage
	    all_rec_movies = set()
	    # varables for popularity
	    popular_sum = 0

	    for i, user in enumerate(self.trainset):
	        if i % 500 == 0:
	            print >> sys.stderr, 'recommended for %d users' % i
	        test_movies = self.testset.get(user, {})
	        rec_movies = self.recommend(user)
	        for movie, w in rec_movies:
	            if movie in test_movies:
	                hit += 1
	            all_rec_movies.add(movie)
	            popular_sum += math.log(1 + self.movie_popular[movie])
	        rec_count += N
	        test_count += len(test_movies)

	    precision = hit / (1.0 * rec_count)
	    recall = hit / (1.0 * test_count)
	    coverage = len(all_rec_movies) / (1.0 * self.movie_count)
	    popularity = popular_sum / (1.0 * rec_count)

	    print >> sys.stderr, 'precision=%.4f\trecall=%.4f\tcoverage=%.4f\tpopularity=%.4f' \
	            % (precision, recall, coverage, popularity)

        
    def random_qp(self):
        return [random.random()*0.1 / SQRT_DIMENSION for _ in xrange(DIMENSION)] #就是随机初始化特征，DIMENSAION=64
    def initModel(self,user_items,F):
        for user,item in self.trainset.iteritems():
            if user not in self.u_f:
                self.u_f.setdefault(user,self.random_qp())
            for i in item:
                if i not in self.i_f:
                    self.i_f.setdefault(i,self.random_qp())
                if i not in self.movie_popular:
                    self.movie_popular[i] = 0
                    self.movie_count += 1
                self.movie_popular[i] += 1
        return [self.u_f,self.i_f]
        # N = Iteration, F = latentFactorNum,lam为正则化参数
    def latentFactorModel(self,F,N,alpha,lam):
        [P,Q] = self.initModel(self.trainset,F)
        for step in range(0,N):
            for user in self.trainset:
                for item,result in self.trainset[user].items():
                    eui = result - self.predict(user,item)
                    for f in range(0,F):
                        P[user][f] += alpha*(eui*Q[item][f] - lam * P[user][f])
                        Q[item][f] += alpha*(eui*P[user][f] - lam * Q[item][f])
            alpha *= 0.9
	    	print ('iteration %d' % step)
if __name__ == '__main__':
     ratingfile = 'ml-1m/ratings.dat'
     lfm = LFM()
     lfm.generate_dataset(ratingfile)
     lfm.latentFactorModel(DIMENSION,20,0.02,0.001)
     lfm.evaluate()




lfm.generate_dataset('ml-1m/ratings.dat')