import time
from matplotlib import pyplot as plt
from math import factorial, sqrt
from decimal import Decimal, getcontext

# BBP Algorithm to compute nth digit of PI
def calc_bbp(n):
    pi_approx = 0
    # Iteratively compute each term of the series
    for k in range(n):
        pi_approx += (1/16**k) * ((4/(8*k+1)) - (2/(8*k+4)) - (1/(8*k+5)) - (1/(8*k+6)))
    return int(pi_approx * 16) % 16

# Spigot Algorithm to compute nth digit of PI
def calc_spigot(n):
    pi_digits = [2]
    # Iteratively compute each digit of PI
    for i in range(1, n+1):
        carry_over = 0
        # Adjust each digit in place and carry over the remainder
        for j in reversed(range(len(pi_digits))):
            val = 10 * pi_digits[j] + carry_over
            pi_digits[j] = val // (2*i - 1)
            carry_over = val % (2*i - 1)
        # Handle any remaining carry
        while carry_over > 0:
            pi_digits.insert(0, carry_over % 10)
            carry_over //= 10
    return pi_digits[-1]

# Chudnovsky Algorithm to compute nth digit of PI
def calc_chudnovsky(n):
    getcontext().prec = n+1
    k_val = n // 14
    chud_sum = Decimal(0)
    # Iteratively compute each term of the series
    for i in range(k_val+1):
        numer = (-1)**i * factorial(6*i) * (13591409 + 545140134*i)
        denom = factorial(3*i) * factorial(i)**3 * 640320**(3*i)
        chud_sum += Decimal(numer) / Decimal(denom)
    chud_sum *= Decimal(12)
    result = Decimal(sqrt(10005)) * chud_sum
    return int(result * 10**n) // 10 % 10

# Array of n-values for which we compute PI's digits
n_values = list(range(1, 1001, 10))
bbp_times = []
spigot_times = []
chudnovsky_times = []

# Time the execution of each algorithm for each n
for n in n_values:
    start_t = time.time()
    calc_bbp(n)
    bbp_times.append(time.time() - start_t)

    start_t = time.time()
    calc_spigot(n)
    spigot_times.append(time.time() - start_t)

    start_t = time.time()
    calc_chudnovsky(n)
    chudnovsky_times.append(time.time() - start_t)

# Plot the time taken by each algorithm
plt.plot(n_values, bbp_times, label='BBP formula')
plt.plot(n_values, spigot_times, label='Spigot algorithm')
plt.plot(n_values, chudnovsky_times, label='Chudnovsky algorithm')
plt.xlabel('n')
plt.ylabel('Time (seconds)')
plt.title('Time required to compute digits of PI')
plt.legend()
plt.show()
