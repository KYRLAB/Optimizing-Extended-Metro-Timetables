from scipy import optimize
from scipy.stats import skewnorm
from scipy.misc import derivative
from scipy import special
import time, random
import numpy as np
import matplotlib.pyplot as plt
class MonteCarloSimulation:
    
    def __init__(self):
        self.x = 80
        self.end_time = 180
        self.scale = 10000
        self.result = []
        self.alpha = self.alpha_calculator(90)
        self.rv = skewnorm(self.alpha)
        self.x_low = skewnorm.ppf(0.01, self.alpha)
        self.x_high = skewnorm.ppf(0.99, self.alpha)
        
    def alpha_calculator(self, x):
        self.x = x
        root = optimize.bisect(self.der_skew_pdf, a = -15, b = 15)
        return root
    
    def der_skew_pdf(self, alpha):
        self.x_low = skewnorm.ppf(0.01, alpha)
        self.x_high = skewnorm.ppf(0.99, alpha)
        x = (self.x_low - self.x_high)/(60-self.end_time)*(self.x-60)+self.x_low
        return (7186705221432913*x*np.exp(-x**2/2)*(special.erf((2**(0.5)*alpha*x)/2)/2-1))/9007199254740992+(7186705221432913*2**(1/2)*alpha*np.exp(-(alpha**2*x**2)/2)*np.exp(-x**2/2))/(18014398509481984*np.pi**(1/2))
    
    def f(self,x):
        x = (self.x_low-self.x_high)/(60-self.end_time)*(x-60)+self.x_low
        return self.rv.cdf(x)*self.scale
    
    def f_dir(self, x):
        return derivative(self.f, x, dx=1e-6)
    
    def generate_sample_point(self,num_variables, fixed_elements = None):
        sample_point = fixed_elements.copy() if fixed_elements is not None else []
        if fixed_elements is not None:
            remaining_elements = [random.choice(list(np.arange(60, self.end_time, 0.5))) for val in range(num_variables-len(fixed_elements))]
        else:
            remaining_elements = [random.choice(list(np.arange(60, self.end_time, 0.5))) for val in range(num_variables)]
        sample_point += remaining_elements 
        sample_point = sorted(sample_point)
        return sample_point

    def objective_function(self,x,num_train):
        obj = 0
        for i in range(num_train):
            if i not in [0, num_train-1]:
                temp_i= x[i]
                temp_i_2 = x[i-1]
                obj += (int((self.f(temp_i)-self.f(temp_i_2))))**3*int((temp_i-temp_i_2))**2 
            elif i == 0:  
                obj += (int((self.f(x[i])-self.f(60))))**3*int((x[i]-60))**2
            elif i == num_train-1:
                obj += (int((self.f(self.end_time)-self.f(x[i-1]))))**3*int((self.end_time-x[i-1]))**2
        return obj//(num_train+2)
    
    def multivariate_monte_carlo_optimization(self, station_name, num_samples, num_train, passenger_flow,fixed_time = []):
        self.scale = passenger_flow
        sample_point = self.generate_sample_point(num_train, fixed_time)
        best_value = self.objective_function(sample_point, num_train)
        best_timetable = sample_point
        start_time = time.time()
        for j in range(num_samples-1):
            if (j+1)%10000 == 1:
                print("Sample Number: {num_sample}".format(num_sample = j+1))
                print("Objective Function Value: {best_value}".format(best_value = best_value))
            sample_point = self.generate_sample_point(num_train, fixed_time)
            value = self.objective_function(sample_point, num_train)
            if value < best_value:
                best_value = value
                best_timetable = sample_point
            sample_point = [f for f in sample_point if f not in fixed_time]
        end_time = time.time()
        elapsed_time = end_time - start_time
        self.result.append([station_name,num_samples,passenger_flow,best_timetable,fixed_time])
        print("Best timetable:", best_timetable)
        print("Best objective value:", best_value)
        print("Elapsed time(s):", elapsed_time)
        return best_timetable
    
    def plot_result(self, best_timetable, fixed_time = None, station = None):
        self.end_time = fixed_time[-1]
        x = np.linspace(60,self.end_time,10000)
        fig, ax = plt.subplots()
        fig.set_size_inches((15/2, 5))
        ax.plot(x, self.f(x), 
               color = 'black', lw=5, alpha=0.6, label='passengers')
        if fixed_time is not None:
            best_timetable = [best_sol for best_sol in best_timetable if best_sol not in fixed_time]
        for i in best_timetable:
            ax.vlines(x = i, color = 'b',linestyle = '--',ymin = self.f(60), ymax = self.f(i))
            ax.hlines(y = self.f(i), color = 'b', linestyle='--', xmin = 60, xmax = i)
        for j in fixed_time:
            ax.vlines(x = j, color = 'r',linestyle = '--',ymin = self.f(60), ymax = self.f(j))
            ax.hlines(y = self.f(j), color = 'r', linestyle='--', xmin = 60, xmax = j)
        ax.set_xlabel('time(min)')
        ax.set_ylabel('passengers(number)')
        ax.set_title('{station} Number of Passenger'.format(station = station))
    