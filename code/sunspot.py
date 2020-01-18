import csv
import numpy as np
import matplotlib.pyplot as plt

def readdataset(filename):
    """readdataset - reads the sunspot data set from the file with filename.
       Returns tuple of (X, t)."""
    X_str = []
    t_str = []
    with open(filename, newline='') as datafile:
        data_reader = csv.reader(datafile, delimiter=' ')
        for row in data_reader:
            X_str.append(row[0:5])
            t_str.append(row[5])

    X = np.array(X_str, dtype=np.float)
    t = np.array(t_str, dtype=np.float)
    return X, t

def factorial_scalar(n):
    """This auxillary function allows for computing factorials of non-integers"""
    return np.prod(np.arange(1, np.math.floor(n)+1, dtype=np.uint64))

def factorial(n_vec):
    """This auxillary function allows for computing factorials of vectors non-integers"""
    f = np.vectorize(factorial_scalar)
    return f(n_vec)

def log_factorial_scalar(n):
    """This auxillary function allows for computing the logarithm of factorials of non-integers
       using Stirling's approximation."""
    n_floor = np.math.floor(n)
    if n_floor == 0:
        return 0.0
    else:
        return n_floor * np.log(n_floor) - n_floor  + 0.5 * np.log(2 * np.pi * n_floor)

def log_factorial(n_vec):
    """This auxillary function allows for computing logarithm of factorials of vectors of non-integers"""
    f = np.vectorize(log_factorial_scalar)
    return f(n_vec)


# Read the training set
X_train, t_train = readdataset('../data/sunspotsTrainStatML.dt')

N_train, D = X_train.shape
print("Training set has X dimension D = " + str(D) + " and N = " + str(N_train) + ' samples.')


# Read the test set
X_test, t_test = readdataset('../data/sunspotsTestStatML.dt')

N_test, D_test = X_test.shape
print("Test set has X dimension D = " + str(D_test) + " and N = " + str(N_test) + ' samples.')


# Visualize the data set
"""
plt.figure()
plt.plot(X_train[:,4], t_train, 'o')
plt.title("Train")
plt.xlabel('X[4]')
plt.ylabel('t')

plt.figure()
plt.plot(X_test[:,4], t_test, 'o')
plt.title("Test")
plt.xlabel('X[4]')
plt.ylabel('t')

plt.figure()
plt.hist(t_train)
plt.title('Train')
plt.xlabel('t values')
plt.ylabel('hist(t)')

plt.figure()
plt.hist(t_test)
plt.title('Test')
plt.xlabel('t values')
plt.ylabel('hist(t)')
"""


#####################################################
# Add your solution here
# It is alright to create additional Python scripts,
# if you find this appropriate.
#####################################################

class Metropolis_Hasting:
    def __init__(self, X, t, iterations, variance = 1.0):
        self.X = self.prepare_data(X, axis = 1)
        self.t = t
        self.mu = self.prepare_data(np.mean(X, axis = 0), axis = 0)
        self.accepted = []
        self.iterations = iterations
        self.variance = variance
        self.run_algorithm()
        self.accepted = self.burn_in(np.array(self.accepted))
        
    def burn_in(self, w, percentage = 0.20):
        quantity = int(percentage * float(len(w)))
        return w[quantity:, :]
    
    def prepare_data(self, data, axis = 1):
        new_data = np.insert(data, 0, 1, axis = axis)
        return new_data
    
    def f(self, x, w):
        return np.matmul(w.T, x)
    
    def prior(self):
        rerun = False
        sample = np.random.multivariate_normal(np.ones(len(self.mu)), 0.25 * np.identity(len(self.mu)))
        
        for x in self.X:
            if (not (self.f(x, sample) > 0)):
                rerun = True
                
        if (not rerun):
            return sample
        else:
            return self.proposal()
    
    def proposal(self):
        rerun = False
        sample = np.random.multivariate_normal(self.mu, self.variance * np.identity(len(self.mu)))
        
        for x in self.X:
            if (not (self.f(x, sample) > 0)):
                rerun = True
                
        if (not rerun):
            return sample
        else:
            return self.proposal()
        
    def likelihood(self, w):
        p = 1
        for i in range(len(self.t)):
            p_temp = np.matmul(np.power(self.f(self.X[i], w), self.t[i]), np.exp(-1 * self.f(self.X[i], w)))
            p_temp /= factorial_scalar(self.t[i])
            p *= p_temp
            
        return p
        
    def log_posterior(self, w):
        s = 0
        for i in range(len(self.t)):
            s += self.t[i] * np.log(self.f(self.X[i], w)) - self.f(self.X[i], w) - factorial_scalar(self.t[i])
        
        s -= 1/(2 * self.variance) * np.matmul(np.subtract(w, self.mu).T, np.subtract(w, self.mu))
        s -= np.log(np.power(np.sqrt(2 * np.pi * self.variance), len(self.mu)))
        # needs to subtract log p(t|X)
        return s
        
    def acceptance(self, w_new, w_t1):
        left_side = 0
        right_side = np.log(self.log_posterior(w_new)/ self.log_posterior(w_t1))
        return min(left_side, right_side)
        
    def run_algorithm(self, autocorrelation = 5):
        w_t1 = self.prior()
        steps = 0
        for i in range(self.iterations):
            steps += 1
            w_new = self.proposal()
            r = self.acceptance(w_new, w_t1)
            u = np.random.normal(0, 1)
            if (r > u):
                if (steps % autocorrelation == 0):
                    self.accepted.append(w_new)
                    w_t1 = w_new
            else:
                if (steps % autocorrelation == 0):
                    self.accepted.append(w_t1)
    
    
MH = Metropolis_Hasting(X_train, t_train, 10000)
MH_mean = np.mean(MH.accepted)
print(MH_mean)

# Show all figures
plt.show()
