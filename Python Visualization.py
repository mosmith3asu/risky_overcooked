from risky_overcooked_py.agents.benchmarking import AgentEvaluator
from risky_overcooked_py.agents.agent import AgentPair, FixedPlanAgent, GreedyHumanModel, RandomAgent, SampleAgent
from risky_overcooked_py.visualization.state_visualizer import StateVisualizer
# from IPython.display import display, Image
from matplotlib.widgets import Button, Slider
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

T = 100
LAYOUT = "sanity_check_3_onion"
ae = AgentEvaluator.from_layout_name({"layout_name": LAYOUT}, {"horizon": T })
trajs = ae.evaluate_human_model_pair()
# print(ae.env)
# print(ae.mdp.state_string)

# action_probs = [ [RandomAgent(all_actions=True).action(state)[1]["action_probs"]]*2 for state in trajs["ep_states"][0]]
# StateVisualizer().display_rendered_trajectory(trajs, ipython_display=False))
path2imgs = StateVisualizer().display_rendered_trajectory(trajs, ipython_display=False)

episode = 0
t = T -1
print(trajs['ep_infos'][episode][t])

# List all files in directory

fnames = os.listdir(path2imgs)

# Load images into a list

imgs = [np.asarray(Image.open(os.path.join(path2imgs, fname))) for fname in fnames]
# Display images


fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
ax.set_xticks([])
ax.set_yticks([])

# axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
axfreq = fig.add_axes([0.125, 0.1, 0.9-0.125, 0.03])
time_slider = Slider(
    ax=axfreq,
    label='t',
    valmin=0,
    valmax=len(imgs)-1,
    valinit=0,
)
img = ax.imshow(imgs[0])


def update(val):
    t =int(time_slider.val)
    infos = ["",""]
    onion_slips = trajs['ep_infos'][0][99]['episode']['ep_game_stats']['onion_slip']
    empty_slips = trajs['ep_infos'][0][99]['episode']['ep_game_stats']['empty_slip']

    for player_idx in range(2):
        if t in onion_slips[player_idx]:
            infos[player_idx] += "Onion Slip"
        elif t in empty_slips[player_idx]:
            infos[player_idx] += "Empty Slip"
    ax.set_title(infos[0] + " | " + infos[1])
    img.set_data(imgs[t])
    # line.set_ydata(f(t, amp_slider.val, freq_slider.val))
    fig.canvas.draw_idle()

time_slider.on_changed(update)
plt.show()