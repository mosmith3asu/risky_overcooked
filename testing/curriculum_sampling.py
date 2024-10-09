import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math

cirriculum_step_threshs = {
    'deliver_soup': 80,
    'pick_up_soup': 80,
    'pick_up_dish': 70,
    'wait_to_cook': 50,
    'deliver_onion3': 50,
    'pick_up_onion3': 50,
    'deliver_onion2': 40,
    'pick_up_onion2': 40,
    'deliver_onion1': 40,
    'full_task': 999
}
cirriculums = list(cirriculum_step_threshs.keys())


def sample_curriculum_step(curriculum_step ):
    n_curr = len(cirriculums)-1
    mu = curriculum_step
    variance = 0.5
    max_deviation = 3
    sigma = math.sqrt(variance)
    # x = np.linspace(mu - max_deviation*sigma, mu + max_deviation*sigma, 100)
    x = np.linspace(min(0,mu - max_deviation * sigma), min(n_curr,mu + max_deviation * sigma), 100)
    p_samples = stats.norm.pdf(x, mu, sigma)
    p_samples = p_samples / np.sum(p_samples)
    xi = np.random.choice(x, p=p_samples)
    return xi


mu = 10
variance = 0.5
max_deviation = 3
sigma = math.sqrt(variance)
# x = np.linspace(mu - max_deviation*sigma, mu + max_deviation*sigma, 100)
x = np.linspace(mu - max_deviation*sigma, mu + max_deviation*sigma, 100)

plt.plot(x, stats.norm.pdf(x, mu, sigma))
#
# p_samples = stats.norm.pdf(x, mu, sigma)
# p_samples = p_samples / np.sum(p_samples)
xi = [sample_cirriculum_step(10) for _ in range(1000)]


# yi = [stats.norm.pdf(xii, mu, sigma) for xii in xi]
# plt.scatter(xi, yi, color='red')

density = stats.kde.gaussian_kde(xi)
x = np.arange(-3, 12, .1)
plt.plot(x, density(x))
plt.show()