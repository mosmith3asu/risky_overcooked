import numpy as np
import matplotlib.pyplot as plt
from risky_overcooked_rl.algorithms.DDQN.eval.policy_heatmap import PolicyHeatmap
from risky_overcooked_rl.utils.model_manager import get_absolute_save_dir
import os

class CompiledModelPreview:
    def __init__(self, layout, p_slip,n_trials=5,seed=42):

        self.layout = layout
        self.p_slip = p_slip
        self.items = ('onion','dish')
        self.dir = get_absolute_save_dir(path='\\risky_overcooked_rl\\algorithms\\DDQN\\models\\')

        hms = {
            'Averse': None,
            'Rational': None,
            'Seeking': None
        }
        titles = []
        # Populate the heatmap data
        print(f'Running {layout} with p_slip={p_slip}')
        for human_type in hms.keys():
            hms[human_type] = PolicyHeatmap(layout, p_slip, human_type=human_type, n_trials=n_trials)
            hms[human_type].run(seed=seed)
            titles = f'{human_type} {self.layout} | p_slip={self.p_slip}'

            print(f'\t | Completed: {human_type}')

        # Create a figure and axis

        print(f'Plotting {layout} with p_slip={p_slip}')
        fig, axs = plt.subplots(3, 3, figsize=(15, 15))
        for r, human_type in enumerate(list(hms.keys())):
            titles = self.items if r == 0 else ['' for _ in range(len(self.items))]
            hms[human_type].plot(axs=list(axs[r, 1:3]), items=self.items, titles=titles)

            print(f'\t | Completed: {human_type}')


        # Add training result figure
        c = 0
        for r, human_type in enumerate(list(hms.keys())):
            self.plot_training_fig(axs[r, c], hms[human_type].policy_fnames[human_type])
            axs[r, c].set_ylabel(human_type)

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
        # img = img[0:430, 0:600, :] # crop image

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
    CMP = CompiledModelPreview(layout='risky_tree2', p_slip=0.2)
    plt.ioff()
    plt.show()

    # # load all images in this directory
    # import os
    # import glob
    # import cv2
    # import numpy as np
    # import matplotlib.pyplot as plt
    # from PIL import Image

    # # Get all image files in the directory
    # image_files = glob.glob(os.path.join(os.path.dirname(__file__), '*.png'))
    # images = []
    # for image_file in image_files:
    #     # Read the image using OpenCV
    #     image = cv2.imread(image_file)
    #     # Convert BGR to RGB
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     images.append(image)
    #
    # # Create a figure and axis
    # fig, ax = plt.subplots()
    # # Loop through the images and display them
    # for i, image in enumerate(images):
    #     # Create a subplot for each image
    #     ax = fig.add_subplot(1, len(images), i + 1)
    #     ax.imshow(image)
    #     ax.axis('off')  # Hide the axis
    #     # Set the title for each subplot
    #     # ax.set_title(f'Image {i + 1}')
    # plt.show()


def subfun():
    pass


if __name__ == "__main__":
    main()
