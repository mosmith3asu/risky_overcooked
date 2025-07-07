import numpy as np
import matplotlib.pyplot as plt
from risky_overcooked_rl.algorithms.DDQN.eval.policy_heatmap import PolicyHeatmap
from risky_overcooked_rl.algorithms.DDQN import get_absolute_save_dir
import os

class CompiledModelPreview:
    def __init__(self, layout, p_slip,n_trials=50,seed=42,overwrite_dict=None,agents=('Averse', 'Rational', 'Seeking')):
        fig_sz = (10, 7)
        self.layout = layout
        self.p_slip = p_slip
        self.items = ('onion','dish')
        self.dir = get_absolute_save_dir()
        hms = {}
        for agent in agents:
            hms[agent] = None
        # hms = {
        #     'Averse': None,
        #     'Rational': None,
        #     'Seeking': None
        # }
        titles = []
        # Populate the heatmap data
        print(f'Running {layout} with p_slip={p_slip}')
        for human_type in hms.keys():
            hms[human_type] = PolicyHeatmap(layout, p_slip, human_type=human_type, robot_type=human_type,
                                            n_trials=n_trials,overwrite_dict=overwrite_dict)
            hms[human_type].run(seed=seed)
            titles = f'{human_type} {self.layout} | p_slip={self.p_slip}'

            print(f'\t | Completed: {human_type}')

        # Create a figure and axis

        print(f'Plotting {layout} with p_slip={p_slip}')
        num_rows = len(hms.keys())
        fig, axs = plt.subplots(num_rows, 3, figsize=fig_sz, constrained_layout=True)
        axs = np.array(axs).reshape(num_rows, 3)  # Ensure axs is a 2D array for easier indexing
        for r, human_type in enumerate(list(hms.keys())):
            titles = self.items if r == 0 else ['' for _ in range(len(self.items))]
            hms[human_type].plot(axs=list(axs[r, 1:3]), items=self.items, titles=titles)

            print(f'\t | Completed: {human_type}')


        # Add training result figure
        c = 0
        for r, human_type in enumerate(list(hms.keys())):
            self.plot_training_fig(axs[r, c], hms[human_type].policy_fnames[human_type])
            # axs[r, c].set_ylabel(human_type)
        #
        for r, human_type in enumerate(list(hms.keys())):
            # axs[r, 0].set_ylabel(human_type)
            axs[r, -1].set_ylabel(human_type)
            axs[r, -1].yaxis.set_label_position("right")

    def plot_training_fig(self,ax,fname,ftype='.png'):

        # select file to load ---------------
        files = os.listdir(path = self.dir)
        files = [f for f in files if (fname in f and ftype in f)]
        if len(files) == 0:
            raise FileNotFoundError(f'No files found with fname:' + fname)
        elif len(files) == 1:
            loads_fname = files[0]
        elif len(files) > 1:
            # warnings.warn(f'Multiple files found with fname: {fname}. Using latest file...')
            loads_fname = files[-1]
        else:
            raise ValueError('Unexpected error occurred')
        PATH = self.dir + loads_fname

        img = plt.imread(PATH)
        img = img[0:430, 0:600, :] # crop image

        # Display the image
        ax.imshow(img)
        ax.axis('off')  # Hide the axis



    def save(self):
        # Save the model preview
        pass

    def load(self):
        # Load the model preview
        pass

    def plot(self):
        # Plot the model preview
        pass


def main():
    # CMP = CompiledModelPreview(layout='risky_tree7', p_slip=0.4)
    # CMP = CompiledModelPreview(layout='risky_tree', p_slip=0.3)
    # CMP = CompiledModelPreview(layout='risky_handoff', p_slip=0.25)
    # CMP = CompiledModelPreview(layout='risky_roundabout', p_slip=0.25)
    # CMP = CompiledModelPreview(layout='risky_mixed_coordination', p_slip=0.2)

    CMP = CompiledModelPreview(layout='risky_mixed_coordination', p_slip=0.5, agents=('Seeking', 'Rational'))
    # CMP = CompiledModelPreview(layout='risky_multipath7', p_slip=0.25,agents=('Averse','Rational', 'Seeking'))

    # CMP = CompiledModelPreview(layout='risky_multipath7', p_slip=0.25,agents=('Averse','Rational', 'Seeking'))

    plt.ioff()
    plt.show()




if __name__ == "__main__":
    main()
