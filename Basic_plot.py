import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp

# linearly spaced numbers
range = 3
steps= 500

x = np.linspace(-range,range,steps)

#Exact solion as function

def y_exact(o ,e):

    numer = sp.jv(1/6,8/(3*e))*sp.jv(-1/6,(o**3)/(3*e)) - sp.jv(-1/6,8/(3*e))*sp.jv(1/6,(o**3)/(3*e))#numerator

    denom = sp.jv(-1/6,1/(3*e))*sp.jv(1/6,8/(3*e)) - sp.jv(-1/6,8/(3*e))*sp.jv(1/6,1/(3*e))

    D = numer/denom

    y = (o**0.5)*(D)

    return y

#WKB approximatiation

def wbk(e, o):
    p = e

    i = complex(0,1)

    BIG = (1-i)/(np.sqrt(2)*(np.exp((16*i)/(3*p)) - np.exp((2*i)/(3*p))))

    c_1 = -np.exp(i/(3*p))*BIG

    c_2 = np.exp((17*i)/(3*p))*BIG

    coef = complex(1,1)/np.sqrt(2)

    exponent = complex(0,1)*(o**3)/(p*3)

    expon_1 = np.exp(exponent)

    expon_2 = np.exp(-exponent)

    y = coef*c_1*(1/o)*expon_1 + coef*c_2*(1/o)*expon_2

    return y




#Bessel function

eps =0.5

y_ex = y_exact(e=eps, o=x)

y_wbk = wbk(e=eps, o=x)

# relative error

rel_error = (y_wbk -y_ex)/y_ex


# setting the axes at the centre
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

plt.xlim([0, range])
plt.ylim([-2, 2])


plt.plot(x,y_ex,'black', label='Exact solution' + ' with epsilon = ' + str(eps))
plt.plot(x,y_wbk,'blue', label='WKB solution' + ' with epsilon = ' + str(eps))

plt.legend(loc='upper left')

# show the plot
plt.show()

