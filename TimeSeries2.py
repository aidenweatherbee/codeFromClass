#"""
#1.generate two synthetic time series, each with at least five sinusoids of different frequencies amplitudes, and phases. Please ensure that the frequency band of the sinusoids in each series is at least four octaves wide and that the ratio of the largest to the lowest amplitude in each series is at least 10. The length of each series should be longer than the largest period used to create the series.
#2. add a Linear trend in the series.
#3.add random noise to the series. use a noise level that is between the two smallest amplitudes used in the simulation.
#4. write custom functions based on the equation used to calculate the mean and variance of a time series made in the code above
#5. write custom functions to calculate the quadratic norm and power of each time series. use it on our two generated time series. 
#6. write custom functions to calculate the auto-covariance and autocorrelation function of each time series. use it on our two generated time series.
#7. write custom functions to calculate the cross-covariance and cross-correlation of the time series. use it on our two generated time series.
#8. plot all of the data generated in steps 1 through 7.
#"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fft



# create a time series
fs = 2000 # sampling rate
t = np.arange(0,20,20/fs) # time vector

# frequencies
f1 = 50
f2 = 5
f3 = 82
f4 = 20
f5 = 30
f6 = 40
f7 = 80
f8 = 50
f9 = 10
f10 = 3

# amplitudes
a1 = 10
a2 = 13
a3 = 4
a4 = 17
a5 = 1
a6 = 22
a7 = 14
a8 = 7
a9 = 18
a10 = 2

# phases

p1 = (np.pi)/6
p2 = (np.pi)/5
p3 = (np.pi)/4
p4 = (np.pi)/3
p5 = (np.pi)/2
p6 = (np.pi)
p7 = (np.pi)*1.3
p8 = (np.pi)*1.6
p9 = (np.pi)*1.8
p10 = (np.pi)*2

#generate sinusoids for first series:
s1 = a1*np.sin(2*np.pi*f1*t + p1)
s2 = a2*np.sin(2*np.pi*f2*t + p2)
s3 = a3*np.sin(2*np.pi*f3*t + p3)
s4 = a4*np.sin(2*np.pi*f4*t + p4)
s5 = a5*np.sin(2*np.pi*f5*t + p5)

#generate sinusoids for second series:
s6 = a6*np.sin(2*np.pi*f6*t + p6)
s7 = a7*np.sin(2*np.pi*f7*t + p7)
s8 = a8*np.sin(2*np.pi*f8*t + p8)
s9 = a9*np.sin(2*np.pi*f9*t + p9)
s10 = a10*np.sin(2*np.pi*f10*t + p10)

# create a linear trend
m = 3 # slope of the line
b = 0 # y-intercept of the line
l = m*t + b # linear trend

# create second linear trend
m = 6 # slope of the line
b = 1 # y-intercept of the line
l2 = m*t + b # linear trend

#generate random noise between smallest amplitudes from above
noise1 = 2*np.random.randn(len(t))#mean of 2, standard deviation of 1
noise2 = 3*np.random.randn(len(t)) #mean of 3, standard deviation of 1
#generate time series 1
ts1 = s1 + s2 + s3 + s4 + s5 + noise1

#generate time series 2
ts2 = s6 + s7 + s8 + s9 + s10 + noise2



#define function to calculate mean
def mean(ts):
    return np.sum(ts)/len(ts)

#define function to calculate variance
def var(ts):
    return np.sum((ts-mean(ts))**2)/len(ts)


#define function to calculate quadratic norm
def quad_norm(ts):
    return np.sqrt(np.sum(ts**2))

#define function to calculate power
def power(ts):
    return np.sum(ts**2)/len(ts)


#define function to calculate auto-covariance
def auto_cov(ts):
    return np.sum((ts-mean(ts))*(ts-mean(ts)))/len(ts)

#define function to calculate auto-correlation function
def auto_corr(ts):
    numerator = np.array([(ts[:len(ts) - tau] * ts[tau:]).sum() for tau in range(len(ts))])

    # Calculate the denominator of the autocorrelation function
    denominator = np.array([(ts - mean(ts)) ** 2]).sum()

    # Calculate the autocorrelation function
    autocorrelation = (numerator / denominator) -1
    return autocorrelation
    


#define function to calculate cross-covariance
def cross_cov(ts1,ts2):
    return np.sum((ts1-mean(ts1))*(ts2-mean(ts2)))/len(ts1)

#define function to calculate cross-correlation
def cross_corr(ts1,ts2):
    return np.sum((ts1-mean(ts1))*(ts2-mean(ts2)))/(np.sqrt(np.sum((ts1-mean(ts1))**2))*np.sqrt(np.sum((ts2-mean(ts2))**2)))


#calculate mean of ts1
mean_ts1 = mean(ts1)
np_mean_ts1 = np.mean(ts1)
print('My code for Mean of ts1:',mean_ts1)
print('numpy code for mean of ts1', np_mean_ts1)

#calculate mean of ts2
mean_ts2 = mean(ts2)
np_mean_ts2 = np.mean(ts2)
print('My code for Mean of ts2:',mean_ts2)
print('Numpy code for Mean of ts2', np_mean_ts2)


#calculate variance of ts1
var_ts1 = var(ts1)
np_var_ts1 = np.var(ts1)
print('My code for Variance of ts1:',var_ts1)
print('Numpy code for Variance of ts1', np_var_ts1)

#calculate variance of ts2
var_ts2 = var(ts2)
np_var_ts2 = np.var(ts2)
print('My code for Variance of ts2:',var_ts2)
print('Numpy code for Variance of ts2', np_var_ts2)


#calculate quadratic norm of ts1
quad_norm_ts1 = quad_norm(ts1)
np_quad_norm_ts1 = np.linalg.norm(ts1)
print('My code for Quadratic norm of ts1:',quad_norm_ts1)
print('Numpy code for Quadratic norm of ts2', np_quad_norm_ts1)


#calculate quadratic norm of ts2
quad_norm_ts2 = quad_norm(ts2)
np_quad_norm_ts2 = np.linalg.norm(ts2)
print('My code for Quadratic norm of ts2:',quad_norm_ts2)
print('Numpy code for Quadratic norm of ts2', np_quad_norm_ts2)

#calculate power of ts1
power_ts1 = power(ts1)
print('My code for Power of ts1:',power_ts1)

#calculate power of ts2
power_ts2 = power(ts2)
print('My code for Power of ts2:',power_ts2)


#calculate auto covariance of ts1
auto_cov_ts1 = auto_cov(ts1)
np_auto_cov_ts1 = np.cov(ts1)
print('My code for Auto covariance of ts1:',auto_cov_ts1)
print('Numpy code for Auto covariance of ts1:',np_auto_cov_ts1)

#calculate auto covariance of ts2
auto_cov_ts2 = auto_cov(ts2)
np_auto_cov_ts2 = np.cov(ts2)
print('My code for Auto covariance of ts1:',auto_cov_ts2)
print('Numpy code for Auto covariance of ts1:',np_auto_cov_ts2)

#calculate auto correlation of ts1
auto_corr_ts1 = auto_corr(ts1)

#calculate auto correlation of ts2
auto_corr_ts2 = auto_corr(ts2)

#calculate cross covariance of ts1 and ts2
cross_cov_ts1_ts2 = cross_cov(ts1,ts2)
np_cross_cov_ts1_ts2 = np.cov(ts1,ts2)
print('My code for Cross covariance of ts1 and ts2:',cross_cov_ts1_ts2)
print('Numpy code for Cross covariance of ts1 and ts2:',np_cross_cov_ts1_ts2)

#calculate cross correlation of ts1 and ts2
cross_corr_ts1_ts2 = cross_corr(ts1,ts2)
np_cross_corr_ts1_ts2 = np.corrcoef(ts1,ts2)
print('My code for Cross correlation of ts1 and ts2:',cross_corr_ts1_ts2)
print('Numpy code for Cross correlation of ts1 and ts2:',np_cross_corr_ts1_ts2)

# plot the signal in the frequency domain
X = fft.fft(ts1) # compute DFT using FFT
N = len(ts1) # length of the signal
k = np.arange(N) # create a vector from 0 to N-1
T = N/fs # get the frequency resolution
frq = k/T # two sides frequency range
frq = frq[range(int(N/2))] # one side frequency range

X1 = fft.fft(ts2) # compute DFT using FFT
N1 = len(ts2) # length of the signal
k1 = np.arange(N1) # create a vector from 0 to N-1
T1 = N1/fs # get the frequency resolution
frq1 = k1/T1 # two sides frequency range
frq1 = frq1[range(int(N1/2))] # one side frequency range


# compute the two-sided spectrum P2. Then compute the single-sided spectrum P1 based on P2 and the even-valued signal length L.
P2 = np.abs(X/N) # two-sided spectrum
P1 = P2[range(int(N/2))] # single-sided spectrum

P4 = np.abs(X1/N1) # two-sided spectrum
P3 = P4[range(int(N1/2))] # single-sided spectrum

#save time series 1
#np.savetxt('C:\\Users\\Aiden\\Desktop\\Schoolwork\\ESSE4020\\Assignment1\\DoubleLength\\ts1.txt', ts1)

#save time series 2
#np.savetxt('C:\\Users\\Aiden\\Desktop\\Schoolwork\\ESSE4020\\Assignment1\\DoubleLength\\ts2.txt', ts2)

#save sinusoid 1
#np.savetxt('C:\\Users\\Aiden\\Desktop\\Schoolwork\\ESSE4020\\Assignment1\\DoubleLength\\s1.txt', s1)

#save sinusoid 2
#np.savetxt('C:\\Users\\Aiden\\Desktop\\Schoolwork\\ESSE4020\\Assignment1\\DoubleLength\\s2.txt', s2)

#save sinusoid 3
#np.savetxt('C:\\Users\\Aiden\\Desktop\\Schoolwork\\ESSE4020\\Assignment1\\DoubleLength\\s3.txt', s3)

#save sinusoid 4
#np.savetxt('C:\\Users\\Aiden\\Desktop\\Schoolwork\\ESSE4020\\Assignment1\\DoubleLength\\s4.txt', s4)

#save sinusoid 5
#np.savetxt('C:\\Users\\Aiden\\Desktop\\Schoolwork\\ESSE4020\\Assignment1\\DoubleLength\\s5.txt', s5)

#save sinusoid 6
#np.savetxt('C:\\Users\\Aiden\\Desktop\\Schoolwork\\ESSE4020\\Assignment1\\DoubleLength\\s6.txt', s6)

#save sinusoid 7
#np.savetxt('C:\\Users\\Aiden\\Desktop\\Schoolwork\\ESSE4020\\Assignment1\\DoubleLength\\s7.txt', s7)

#save sinusoid 8
#np.savetxt('C:\\Users\\Aiden\\Desktop\\Schoolwork\\ESSE4020\\Assignment1\\DoubleLength\\s8.txt', s8)

#save sinusoid 9
#np.savetxt('C:\\Users\\Aiden\\Desktop\\Schoolwork\\ESSE4020\\Assignment1\\DoubleLength\\s9.txt', s9)

#save sinusoid 10
#np.savetxt('C:\\Users\\Aiden\\Desktop\\Schoolwork\\ESSE4020\\Assignment1\\DoubleLength\\s10.txt', s10)

# save ranmdom noise1
#np.savetxt('C:\\Users\\Aiden\\Desktop\\Schoolwork\\ESSE4020\\Assignment1\\DoubleLength\\noise1.txt', noise1)

#np.savetxt('C:\\Users\\Aiden\\Desktop\\Schoolwork\\ESSE4020\\Assignment1\\DoubleLength\\noise2.txt', noise2)

#plot time series 1
plt.figure(1)
plt.plot(t,ts1)
plt.title('Time Series 1')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

#plot time series 1
plt.figure(2)
plt.plot(t,auto_corr_ts1)
plt.title('Time Series 1 ACF')
plt.xlabel('Time (s)')
plt.ylabel('Auto Correlation Function(Lag)')

#plot time series 2
plt.figure(3)
plt.plot(t,ts2)
plt.title('Time Series 2')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

#plot time series 2
plt.figure(4)
plt.plot(t,auto_corr_ts2)
plt.title('Time Series 2 ACF')
plt.xlabel('Time (s)')
plt.ylabel('Auto Correlation Function (Lag)')

plt.figure(5)
plt.title('Spectrum Time Series 1')
plt.plot(frq,P1) # plotting the spectrum
plt.xlabel('Freq (Hz)')
plt.ylabel('|P1(freq)|')

plt.figure(6)
plt.title('Spectrum Time Series 2')
plt.plot(frq1,P3) # plotting the spectrum
plt.xlabel('Freq (Hz)')
plt.ylabel('|P1(freq)|')

plt.figure(7)
plt.title('Time series 1 noise')
plt.plot(t, noise1) # plotting the spectrum
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.figure(8)
plt.title('Time series 2 noise')
plt.plot(t, noise2) # plotting the spectrum
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.figure(9)
plt.title('Time Series 1 Sinusoid 1')
plt.plot(t, s1) # plotting the spectrum
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.figure(10)
plt.title('Time Series 1 Sinusoid 2')
plt.plot(t, s2) # plotting the spectrum
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.figure(11)
plt.title('Time Series 1 Sinusoid 3')
plt.plot(t, s3) # plotting the spectrum
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.figure(12)
plt.title('Time Series 1 Sinusoid 4')
plt.plot(t, s4) # plotting the spectrum
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.figure(13)
plt.title('Time Series 1 Sinusoid 5')
plt.plot(t, s5) # plotting the spectrum
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.figure(14)
plt.title('Time Series 2 Sinusoid 6')
plt.plot(t, s6) # plotting the spectrum
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.figure(15)
plt.title('Time Series 2 Sinusoid 7')
plt.plot(t, s7) # plotting the spectrum
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.figure(16)
plt.title('Time Series 2 Sinusoid 8')
plt.plot(t, s8) # plotting the spectrum
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.figure(17)
plt.title('Time Series 2 Sinusoid 9')
plt.plot(t, s9) # plotting the spectrum
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.figure(18)
plt.title('Time Series 2 Sinusoid 10')
plt.plot(t, s10) # plotting the spectrum
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.show()

