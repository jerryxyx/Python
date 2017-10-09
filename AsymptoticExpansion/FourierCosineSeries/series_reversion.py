import scipy as sp

def inverseSeries(coeffs):
    a = coeffs
    inverseCoeffs = []
    inverseCoeffs.append(0)
    inverseCoeffs.append(1/a[1])
    inverseCoeffs.append(-a[1]**(-3)*a[2])
    inverseCoeffs.append(a[1]**(-5) * (2*a[2]**2-a[1]*a[3]))
    inverseCoeffs.append(a[1]**(-7) * (5*a[1]*a[2]*a[3] - a[1]**2*a[4] - 5*a[2]**3))
    inverseCoeffs.append(a[1]**(-9) * (6*a[1]**2*a[2]*a[4] + 3*a[1]**2*a[3]**2
                                       +14*a[2]**4-a[1]**3*a[5] - 21* a[1]*a[2]**2 *a[3]))
    inverseCoeffs.append(a[1]**(-11) * (7*a[1]**3*a[2]*a[5] + 7*a[1]**3*a[3]*a[4]
                                        + 84*a[1]*a[2]**3*a[3] - a[1]**4*a[6]
                                        - 28*a[1]**2*a[2]*a[3]**2 - 42*a[2]**5
                                        - 28*a[1]**2*a[2]**2*a[4]))
    inverseCoeffs.append(a[1]**(-13) * (8*a[1]**4*a[2]*a[6] + 8*a[1]**4*a[3]*a[5]
                                        + 4*a[1]**4*a[4]**2 + 120*a[1]**2*a[2]**3*a[4]
                                        + 180* (a[1]*a[2]*a[3])**2 + 132*a[2]**6
                                        - a[1]**5*a[7] - 36*a[1]**3*a[2]**2*a[5]
                                        - 72*a[1]**3*a[2]*a[3]*a[4] - 12*a[1]**3*a[3]**3
                                        - 330*a[1]*a[2]**4*a[3]))
    return inverseCoeffs
