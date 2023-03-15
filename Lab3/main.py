import math
import time
import matplotlib.pyplot as plt
def sieve_of_eratosthenes_1(n):
    c = [True] * (n+1)
    c[1] = False
    i = 2
    while i <= n:
        if c[i]:
            j = 2 * i
            while j <= n:
                c[j] = False
                j = j + i
        i = i + 1
    return [i for i in range(1, n+1) if c[i]]

def sieve_of_eratosthenes_2(n):
    c = [True] * (n+1)
    c[1] = False
    i = 2
    while i <= n:
        j = 2 * i
        while j <= n:
            c[j] = False
            j = j + i
        i = i + 1
    return [i for i in range(1, n+1) if c[i]]

def sieve_of_eratosthenes_3(n):
    c = [True] * (n+1)
    c[1] = False
    i = 2
    while i <= n:
        if c[i]:
            j = i + 1
            while j <= n:
                if j % i == 0:
                    c[j] = False
                j = j + 1
        i = i + 1
    return [i for i in range(1, n+1) if c[i]]

def sieve_of_eratosthenes_4(n):
    c = [True] * (n+1)
    c[1] = False
    i = 2
    while i <= n:
        j = 2
        while j < i:
            if i % j == 0:
                c[i] = False
            j = j + 1
        i = i + 1
    return [i for i in range(1, n+1) if c[i]]



def sieve_of_eratosthenes_5(n):
    c = [True] * (n+1)
    c[1] = False
    i = 2
    while i <= n:
        j = 2
        while j <= math.sqrt(i):
            if i % j == 0:
                c[i] = False
            j = j + 1
        i = i + 1
    return [i for i in range(1, n+1) if c[i]]

print(sieve_of_eratosthenes_1(10000))
print(sieve_of_eratosthenes_2(10000))
print(sieve_of_eratosthenes_3(10000))
print(sieve_of_eratosthenes_4(10000))
print(sieve_of_eratosthenes_5(10000))


def time_it(func, *args):
    start_time = time.time()
    func(*args)
    end_time = time.time()
    return end_time - start_time

def plot_time_complexity_sep(algorithms, sizes):
    for i, algorithm in enumerate(algorithms):
        times = []
        for size in sizes:
            times.append(time_it(algorithm, size))
        plt.plot(sizes, times, label=f'Algorithm {i+1}')
        plt.legend()
        plt.xlabel('Input Size')
        plt.ylabel('Execution Time (s)')
        plt.show()
        plt.clf()

def plot_time_complexity(algorithms, sizes):
    for i, algorithm in enumerate(algorithms):
        times = []
        for size in sizes:
            times.append(time_it(algorithm, size))
        plt.plot(sizes, times, label=f'Algorithm {i+1}')
    plt.legend()
    plt.xlabel('Input Size')
    plt.ylabel('Execution Time (s)')
    plt.show()

sizes = range(100, 1100, 100)
algorithms = [
    sieve_of_eratosthenes_1,
    sieve_of_eratosthenes_2,
    sieve_of_eratosthenes_3,
    sieve_of_eratosthenes_4,
    sieve_of_eratosthenes_5
]

algorithm_1 = [sieve_of_eratosthenes_1]
algorithm_2 = [sieve_of_eratosthenes_2]
algorithm_3 = [sieve_of_eratosthenes_3]
algorithm_4 = [sieve_of_eratosthenes_4]
algorithm_5 = [sieve_of_eratosthenes_5]
plot_time_complexity(algorithms, sizes)
plot_time_complexity_sep(algorithms, sizes)
