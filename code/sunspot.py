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


# This function has been taken from my answer to assignment 1.
# The assignment was fully made by me and not made in any collaborations.
def RMSE(t, tp):
    N = len(t)
    s = 0
    for i in range(N):
        s += np.linalg.norm((t[i] - tp[i]))**2
    s /= N
    s = s**(0.5)
    return s

class Metropolis_Hasting:
    def __init__(self, X, t, iterations, variance = 0.25):
        self.X = self.prepare_data(X, axis = 1)
        self.t = t
        self.mu = np.ones(self.X.shape[1])
        self.accepted = []
        self.iterations = iterations
        self.variance = variance
        self.fit()
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
        
    def log_posterior(self, w):
        s = 0
        for i in range(len(self.t)):
            s += self.t[i] * np.log(self.f(self.X[i], w)) - self.f(self.X[i], w) - log_factorial_scalar(self.t[i])

        s -= 1/(2 * self.variance) * np.matmul(np.subtract(w, self.mu).T, np.subtract(w, self.mu))
        s -= np.log(np.power(np.sqrt(2 * np.pi * self.variance), len(self.mu)))

        return s
    
    def log_posterior1(self, w):
        s = 0
        for i in range(len(self.t)):
            s += self.t[i] * np.log(self.f(self.X[i], w)) - self.f(self.X[i], w) - log_factorial_scalar(self.t[i])

        s -= 1/(2 * self.variance) * np.matmul(np.subtract(w, self.mu).T, np.subtract(w, self.mu))
        s -= np.log(np.power(np.sqrt(2 * np.pi * self.variance), len(self.mu)))

        return s
        
    def acceptance(self, w_new, w_t1):
        left_side = 0
        right_side = self.log_posterior(w_new) - self.log_posterior1(w_t1)

        return min(left_side, right_side)
        
    def fit(self, autocorrelation = 5):
        steps = 0
        w_t1 = self.prior()
        for i in range(self.iterations):
            steps += 1
            w_new = self.proposal()
            r = self.acceptance(w_new, w_t1)

            u = np.random.normal(0, 1)
            if (r > np.log(u)):
                if (steps % autocorrelation == 0):
                    self.accepted.append(w_new)
                    w_t1 = w_new

    def predict(self, X):
        t = []
        for x in X:
            x = np.array(x)
            s = 0
            for w in self.accepted:
                w = np.array(w)
                s += self.f(self.prepare_data(x, axis = 0), w)
                
            s /= len(self.accepted)
            t.append(s)
        
        t = np.array(t).reshape((-1, 1))
        return t

model_1 = Metropolis_Hasting(X_train[:, 4].reshape((-1, 1)), t_train, 1000)
model_1_predictions = model_1.predict(X_test[:, 4].reshape((-1, 1)))
RMSE_1 = RMSE(t_test, model_1_predictions)
print("RMSE 1: {}".format(RMSE_1))

model_2 = Metropolis_Hasting(X_train[:, 2:4], t_train, 10000)
model_2_predictions = model_2.predict(X_test[:, 2:4])
RMSE_2 = RMSE(t_test, model_2_predictions)
print("RMSE 2: {}".format(RMSE_2))

model_3 = Metropolis_Hasting(X_train, t_train, 10000)
model_3_predictions = model_3.predict(X_test)
RMSE_3 = RMSE(t_test, model_3_predictions)
print("RMSE 3: {}".format(RMSE_3))

#plt.plot(model_1.accepted[:, 0], model_1.accepted[:, 1], 'bo')

# Show all figures
#plt.show()
