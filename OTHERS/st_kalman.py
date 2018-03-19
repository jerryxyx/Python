{\rtf1\ansi\ansicpg1252\cocoartf1348\cocoasubrtf170
{\fonttbl\f0\fmodern\fcharset0 Courier-Bold;\f1\fmodern\fcharset0 Courier;\f2\fmodern\fcharset0 Courier-Oblique;
}
{\colortbl;\red255\green255\blue255;\red90\green174\blue29;\red0\green0\blue0;\red54\green105\blue196;
\red135\green135\blue135;\red197\green197\blue197;\red43\green98\blue152;\red32\green126\blue139;\red231\green140\blue18;
}
\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\sl284

\f0\b\fs22 \cf2 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 mport
\f1\b0 \cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3  \cf4 \expnd0\expndtw0\kerning0
\ul \ulc4 \outl0\strokewidth0 \strokec4 numpy\cf3 \expnd0\expndtw0\kerning0
\ulnone \outl0\strokewidth0 \strokec3 \

\f0\b \cf2 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 import
\f1\b0 \cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3  \cf4 \expnd0\expndtw0\kerning0
\ul \ulc4 \outl0\strokewidth0 \strokec4 pylab\cf3 \expnd0\expndtw0\kerning0
\ulnone \outl0\strokewidth0 \strokec3 \
\
\pard\pardeftab720\sl284

\f2\i \cf5 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec5 # intial parameters
\f1\i0 \cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 \
\pard\pardeftab720\sl284
\cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 n_iter\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3  \cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 =\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3  \cf7 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec7 50\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 \
\cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 sz\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3  \cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 =\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3  \cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 (n_iter,)\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3  
\f2\i \cf5 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec5 # size of array
\f1\i0 \cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 \
\cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 x\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3  \cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 =\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3  \cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 -\cf7 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec7 0.37727\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3  
\f2\i \cf5 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec5 # truth value (typo in example at top of p. 13 calls this z)
\f1\i0 \cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 \
\cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 z\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3  \cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 =\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3  \cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 numpy.random.normal(x,\cf7 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec7 0.1\cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 ,size=sz)\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3  
\f2\i \cf5 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec5 # observations (normal about x, sigma=0.1)
\f1\i0 \cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 \
\
\cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 Q\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3  \cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 =\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3  \cf7 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec7 1e-5\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3  
\f2\i \cf5 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec5 # process variance
\f1\i0 \cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 \
\
\pard\pardeftab720\sl284

\f2\i \cf5 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec5 # allocate space for arrays
\f1\i0 \cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 \
\pard\pardeftab720\sl284
\cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 xhat=numpy.zeros(sz)\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3       
\f2\i \cf5 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec5 # a posteri estimate of x
\f1\i0 \cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 \
\cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 P=numpy.zeros(sz)\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3          
\f2\i \cf5 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec5 # a posteri error estimate
\f1\i0 \cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 \
\cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 xhatminus=numpy.zeros(sz)\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3  
\f2\i \cf5 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec5 # a priori estimate of x
\f1\i0 \cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 \
\cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 Pminus=numpy.zeros(sz)\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3     
\f2\i \cf5 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec5 # a priori error estimate
\f1\i0 \cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 \
\cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 K=numpy.zeros(sz)\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3          
\f2\i \cf5 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec5 # gain or blending factor
\f1\i0 \cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 \
\
\cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 R\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3  \cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 =\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3  \cf7 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec7 0.1\cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 **\cf7 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec7 2\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3  
\f2\i \cf5 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec5 # estimate of measurement variance, change to see effect
\f1\i0 \cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 \
\
\pard\pardeftab720\sl284

\f2\i \cf5 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec5 # intial guesses
\f1\i0 \cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 \
\pard\pardeftab720\sl284
\cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 xhat[\cf7 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec7 0\cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 ]\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3  \cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 =\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3  \cf7 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec7 0.0\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 \
\cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 P[\cf7 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec7 0\cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 ]\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3  \cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 =\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3  \cf7 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec7 1.0\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 \
\
\pard\pardeftab720\sl284

\f0\b \cf2 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 for
\f1\b0 \cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3  \cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 k\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3  
\f0\b \cf2 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 in
\f1\b0 \cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3  \cf8 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec8 range\cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 (\cf7 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec7 1\cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 ,n_iter):\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 \
    
\f2\i \cf5 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec5 # time update
\f1\i0 \cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 \
    \cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 xhatminus[k]\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3  \cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 =\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3  \cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 xhat[k-\cf7 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec7 1\cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 ]\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 \
    \cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 Pminus[k]\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3  \cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 =\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3  \cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 P[k-\cf7 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec7 1\cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 ]+Q\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 \
\
    
\f2\i \cf5 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec5 # measurement update
\f1\i0 \cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 \
    \cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 K[k]\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3  \cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 =\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3  \cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 Pminus[k]/(\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3  \cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 Pminus[k]+R\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3  \cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 )\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 \
    \cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 xhat[k]\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3  \cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 =\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3  \cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 xhatminus[k]+K[k]*(z[k]-xhatminus[k])\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 \
    \cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 P[k]\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3  \cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 =\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3  \cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 (\cf7 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec7 1\cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 -K[k])*Pminus[k]\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 \
\
\pard\pardeftab720\sl284
\cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 pylab.figure()\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 \
\cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 pylab.plot(z,\cf9 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec9 'k+'\cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 ,label=\cf9 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec9 'noisy measurements'\cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 )\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 \
\cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 pylab.plot(xhat,\cf9 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec9 'b-'\cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 ,label=\cf9 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec9 'a posteri estimate'\cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 )\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 \
\cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 pylab.axhline(x,color=\cf9 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec9 'g'\cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 ,label=\cf9 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec9 'truth value'\cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 )\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 \
\cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 pylab.legend()\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 \
\cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 pylab.xlabel(\cf9 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec9 'Iteration'\cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 )\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 \
\cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 pylab.ylabel(\cf9 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec9 'Voltage'\cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 )\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 \
\
\cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 pylab.figure()\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 \
\cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 valid_iter\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3  \cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 =\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3  \cf8 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec8 range\cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 (\cf7 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec7 1\cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 ,n_iter)\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3  
\f2\i \cf5 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec5 # Pminus not valid at step 0
\f1\i0 \cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 \
\cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 pylab.plot(valid_iter,Pminus[valid_iter],label=\cf9 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec9 'a priori error estimate'\cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 )\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 \
\cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 pylab.xlabel(\cf9 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec9 'Iteration'\cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 )\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 \
\cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 pylab.ylabel(\cf9 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec9 '$(Voltage)^2$'\cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 )\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 \
\cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 pylab.setp(pylab.gca(),\cf9 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec9 'ylim'\cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 ,[\cf7 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec7 0\cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 ,.\cf7 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec7 01\cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 ])\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 \
\cf6 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec6 pylab.show()\cf3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 \
}