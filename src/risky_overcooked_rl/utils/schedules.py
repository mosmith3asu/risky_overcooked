from abc import abstractmethod

import numpy as np
import matplotlib.pyplot as plt





#
class Schedule:
    def __init__(self,start,end,duration,type, decay=1, total_iterations = None):
        self.start = start
        self.end = end
        self.duration = duration
        self.decay = decay
        self.iter = 0
        self.type = type
        self.current_val = start

        # duration given in percent
        if duration <= 1:
            assert total_iterations is not None, "Total iterations must be provided for percentage duration"
            duration = int(total_iterations * duration)

        # Init Schedules
        if start == end:
            self.schedule = np.ones(total_iterations) * start

        elif type.lower() == 'linear' or type.lower() == 'lin':
            self.schedule = np.linspace(start, end, duration)

        elif type.lower() == 'exponential' or type.lower() == 'exp':
            iters = np.arange(0, duration)
            # iters = np.arange(0, total_iterations)
            self.schedule = start * (end / start) ** ((iters / duration) ** (1 / decay))
        else:
            raise ValueError(f"Unknown schedule type: {type}. Supported types are 'linear' or 'exponential'.")



    def step(self, t=None):
        """ If none, use internal iteration count"""
        if t is None:
            t = self.iter
            self.iter += 1
        self.current_val = self.schedule[t] if t < len(self.schedule) else self.end
        return self.current_val

    def preview(self, T=None):
        """ Preview the schedule"""
        T = self.duration if T is None else T
        vals = [self.step(t=t) for t in range(T)]
        plt.plot(vals)
        plt.title(f'{self.type} Schedule')
        plt.xlabel('Iterations')
        plt.ylabel('Value')
        plt.ioff()
        plt.show()



if __name__ == "__main__":
    lin_sched = {
        'start': 1,
        'end': 0,
        'duration': 1_000,
        'type': 'linear',
    }
    exp_sched = {
        'start': 1.0,
        'end': 0.05,
        'duration': 5_000,
        'decay': 1,
        'type': 'exponential',
    }

    # Schedule(**lin_sched).preview(1500)
    Schedule(**exp_sched).preview(5500)
