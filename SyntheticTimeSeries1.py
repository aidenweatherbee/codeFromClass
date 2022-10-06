#"""
# use python to generate a synthetic time series. the series should contain:
#1. At least five  sinusoids of different frequencies amplitudes, and phases. Please ensure that the frequency band of the sinusoids is at least four octaves wide and that the ratio of the largest to the lowest amplitude is at least 10. The length of the series should be slightly longer than the largest period used to create the series.
#2) should also contain a Linear tren in the synthetic data
#3) should also contain random noise. use a noise level that is at a maximum, between the two smallest amplitudes used in the simulation.
#"""

import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.fftpack as fft

# create a time series
fs = 1000 # sampling rate
t = np.arange(0,10,10/fs) # time vector

# create a sinusoid
f1 =  random.randint(1,10)# frequency
a1 = 1 # amplitude
phi1 = 0 # phase
s1 = a1*np.sin(2*np.pi*f1*t + phi1) # sinusoid

# create a second sinusoid
f2 = 20 # frequency
a2 = 2 # amplitude
phi2 = 0 # phase
s2 = a2*np.sin(2*np.pi*f2*t + phi2) # sinusoid

# create a third sinusoid
f3 = 40 # frequency
a3 = 3 # amplitude
phi3 = 0 # phase
s3 = a3*np.sin(2*np.pi*f3*t + phi3) # sinusoid

# create a fourth sinusoid
f4 = 80 # frequency
a4 = 4 # amplitude
phi4 = 0 # phase
s4 = a4*np.sin(2*np.pi*f4*t + phi4) # sinusoid

# create a fifth sinusoid
f5 = 160 # frequency
a5 = 5 # amplitude
phi5 = 0 # phase
s5 = a5*np.sin(2*np.pi*f5*t + phi5) # sinusoid

# create a sixth sinusoid
f6 = 160 # frequency
a6 = 5 # amplitude
phi6 = 0 # phase
s6 = a6*np.sin(2*np.pi*f6*t + phi6) # sinusoid

# create a linear trend
m = 1 # slope of the line
b = 0 # y-intercept of the line
l = m*t + b # linear trend


# create random noise
n = np.random.randn(len(t)) # random noise


# create the signal
x = s1 + s2 + s3 + s4 + s5 + l + n


# plot the signal in the time domain
plt.figure(1)
plt.clf()
plt.plot(t,x)
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')


# plot the signal in the frequency domain
X = fft.fft(x) # compute DFT using FFT
N = len(x) # length of the signal
k = np.arange(N) # create a vector from 0 to N-1
T = N/fs # get the frequency resolution
frq = k/T # two sides frequency range
frq = frq[range(int(N/2))] # one side frequency range


# compute the two-sided spectrum P2. Then compute the single-sided spectrum P1 based on P2 and the even-valued signal length L.
P2 = np.abs(X/N) # two-sided spectrum
P1 = P2[range(int(N/2))] # single-sided spectrum


plt.figure(2)
plt.clf()
plt.plot(frq,P1) # plotting the spectrum
plt.xlabel('Freq (Hz)')
plt.ylabel('|P1(freq)|')
plt.show()

#"""
#Write python code to calculate:
#1. The Mean and variance of a time series made in the code above
#2. The quadratic norm and power of each time series
#3. The auto-covariance and autocorrelation function of each time series.
#4. The cross-covariance and cross-correlation of the time series
#5. Repeat the above by doubling the length of both series. How do the results compare with the original series?

#"""


#1. The Mean and variance of a time series made in the code above

mean = np.mean(x)
variance = np.var(x)
print("Mean: ", mean)
print("Variance: ", variance)

#2. The quadratic norm and power of each time series

quadratic_norm = np.linalg.norm(x)
power = np.sum(x**2)
print("Quadratic Norm: ", quadratic_norm)
print("Power: ", power)

#3. The auto-covariance and autocorrelation function of each time series.

auto_covariance = np.cov(x)
auto_correlation = np.correlate(x, x, mode='full')
print("Auto-Covariance: ", auto_covariance)
print("Auto-Correlation: ", auto_correlation)

#4. The cross-covariance and cross-correlation of the time series

cross_covariance = np.cov(x, x)
cross_correlation = np.correlate(x, x, mode='full')
print("Cross-Covariance: ", cross_covariance)
print("Cross-Correlation: ", cross_correlation)


#5. Repeat the above by doubling the length of both series. How do the results compare with the original series?

# Doubling the length of the time series
t2 = np.arange(0,20,20/fs) # time vector
x2 = s1 + s2 + s3 + s4 + s5 + l + n # signal


#1. The Mean and variance of a time series made in the code above
mean2 = np.mean(x2)
variance2 = np.var(x2)
print("Mean: ", mean2)
print("Variance: ", variance2)


#2. The quadratic norm and power of each time series
quadratic_norm2 = np.linalg.norm(x2)
power2 = np.sum(x2**2)
print("Quadratic Norm: ", quadratic_norm2)
print("Power: ", power2)


#3. The auto-covariance and autocorrelation function of each time series.
auto_covariance2 = np.cov(x2)
auto_correlation2 = np.correlate(x2, x2, mode='full')
print("Auto-Covariance: ", auto_covariance2)
print("Auto-Correlation: ", auto_correlation2)


#4. The cross-covariance and cross-correlation of the time series
cross_covariance2 = np.cov(x2, x2)
cross_correlation2 = np.correlate(x2, x2, mode='full')
print("Cross-Covariance: ", cross_covariance2)
print("Cross-Correlation: ", cross_correlation2)