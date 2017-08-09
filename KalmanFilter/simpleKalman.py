import numpy
import pylab

# intial parameters
iteration_count = 50
empty_iteration_tuple = (iteration_count,)  # size of array
actual_value = -0.37727  # truth value
noisy_measurement = numpy.random.normal(actual_value, 0.1, size=empty_iteration_tuple)  # observations (normal about x, sigma=0.1)

process_variance = 1e-5  # process variance

# allocate space for arrays
posteri_estimate = numpy.zeros(empty_iteration_tuple)
posteri_error_estimate = numpy.zeros(empty_iteration_tuple)
priori_estimate = numpy.zeros(empty_iteration_tuple)
priori_error_estimate = numpy.zeros(empty_iteration_tuple)
blending_factor = numpy.zeros(empty_iteration_tuple)

estimated_measurement_variance = 0.01  # estimate of measurement variance, change to see effect

# intial guesses
posteri_estimate[0] = 0.0
posteri_error_estimate[0] = 1.0

for iteration in range(1, iteration_count):
    # time update
    priori_estimate[iteration] = posteri_estimate[iteration - 1]
    priori_error_estimate[iteration] = posteri_error_estimate[iteration - 1] + process_variance

    # measurement update
    blending_factor[iteration] = priori_error_estimate[iteration] / (priori_error_estimate[iteration] + estimated_measurement_variance)
    # noisy measurement is the only thing where we need the entire list
    posteri_estimate[iteration] = priori_estimate[iteration] + blending_factor[iteration] * (noisy_measurement[iteration] - priori_estimate[iteration])
    posteri_error_estimate[iteration] = (1 - blending_factor[iteration]) * priori_error_estimate[iteration]

pylab.figure()
pylab.plot(noisy_measurement, 'k+', label='noisy measurements')
pylab.plot(posteri_estimate, 'b-', label='a posteri estimate')
pylab.axhline(actual_value, color='g', label='truth value')
pylab.legend()
pylab.xlabel('Iteration')
pylab.ylabel('Voltage')
pylab.show()