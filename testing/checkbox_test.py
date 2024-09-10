import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
plt.ion()

class CBtest(object):
    def __init__(self):
        self.checkboxes = {
            's1': True,
            'S2': True,
        }

    # def spawn_figure(self):
    #     self.fig, self.ax = plt.subplots()
    #     # DUMMY DATA
    #     t = np.arange(0.0, 2.0, 0.01)
    #     s0 = np.sin(2 * np.pi * t)
    #     s1 = np.sin(4 * np.pi * t)
    #     s2 = np.sin(6 * np.pi * t)
    #     l0, = self.ax.plot(t, s0, visible=False, lw=2, color='k', label='2 Hz')
    #     l1, = self.ax.plot(t, s1, lw=2, color='r', label='4 Hz')
    #     l2, = self.ax.plot(t, s2, lw=2, color='g', label='6 Hz')
    #
    #
    #     plt.subplots_adjust(right=0.7)
    #     # self.fig.close()
    #     self.ax.set_xticks([])
    #     self.ax.set_yticks([])
    #
    #     ax_cb = self.fig.add_axes([0.75, 0.1, 0.2, 0.8])  # (left, bottom, width, height)
    #     check = CheckButtons(
    #         ax=ax_cb,
    #         labels=[key for key in self.checkboxes.keys()],  # self.checkboxes.keys(),
    #         actives=[v for v in self.checkboxes.values()],
    #     )
    #     check.on_clicked(self.cb_callback)
    #
    #     self.fig_number = self.fig.number
    #     return self.fig, self.ax
    #
    # def cb_callback(self, label):
    #     self.checkboxes[label] = not self.checkboxes[label]
    #     print(self.checkboxes)
    #     # self.fig.canvas.draw()
    #     # self.fig.canvas.flush_events()
    #
    # def preview(self):
    #     fig,ax = self.spawn_figure()
    #     # self.fig.show()
    #     while True:
    #         fig.canvas.draw()
    #         fig.canvas.flush_events()
    def spawn_figure(self):
        self.fig, self.ax = plt.subplots()
        # return fig, ax

    def cb_callback(self,label):
        print(label)
        # index = labels.index(label)
        # lines[index].set_visible(not lines[index].get_visible())
        # plt.draw()
    def draw_heatmap(self):
        t = np.arange(0.0, 2.0, 0.01)
        s0 = np.sin(2*np.pi*t)
        s1 = np.sin(4*np.pi*t)
        s2 = np.sin(6*np.pi*t)

        l0, = self.ax.plot(t, s0, visible=False, lw=2, color='k', label='2 Hz')
        l1, = self.ax.plot(t, s1, lw=2, color='r', label='4 Hz')
        l2, = self.ax.plot(t, s2, lw=2, color='g', label='6 Hz')
    def draw_checkboxes(self):
        plt.subplots_adjust(right=0.8)

        # lines = [l0, l1, l2]

        # Make checkbuttons with all plotted lines with correct visibility
        # rax = plt.axes([0.05, 0.4, 0.1, 0.15])
        ax_cb = self.fig.add_axes([0.75, 0.1, 0.2, 0.8])  # (left, bottom, width, height)
        labels = [key for key in self.checkboxes.keys()]
        visibility = [val for val in self.checkboxes.values()]
        check = CheckButtons(ax_cb, labels, visibility)

        check.on_clicked(self.cb_callback)
    def preview(self):
        self.spawn_figure()
        self.draw_heatmap()
        self.draw_checkboxes()
        # fig,ax = self.fig, self.ax #self.spawn_figure()

        # plt.subplots_adjust(right=0.8)
        #
        # # lines = [l0, l1, l2]
        #
        # # Make checkbuttons with all plotted lines with correct visibility
        # # rax = plt.axes([0.05, 0.4, 0.1, 0.15])
        # ax_cb = self.fig.add_axes([0.75, 0.1, 0.2, 0.8])  # (left, bottom, width, height)
        # labels = [key for key in self.checkboxes.keys()]
        # visibility = [val for val in self.checkboxes.values()]
        # check = CheckButtons(ax_cb, labels, visibility)
        #
        # check.on_clicked(self.cb_callback)


        while True:
            self.fig.canvas.flush_events()




cb = CBtest()
cb.preview()
# while True:
#     cb.fig.canvas.draw()
#     cb.fig.canvas.flush_events()

# t = np.arange(0.0, 2.0, 0.01)
# s0 = np.sin(2*np.pi*t)
# s1 = np.sin(4*np.pi*t)
# s2 = np.sin(6*np.pi*t)
#
# checkboxes = {
#     's1': True,
#     'S2': True,
# }
#
# fig, ax = plt.subplots()
# l0, = ax.plot(t, s0, visible=False, lw=2, color='k', label='2 Hz')
# l1, = ax.plot(t, s1, lw=2, color='r', label='4 Hz')
# l2, = ax.plot(t, s2, lw=2, color='g', label='6 Hz')
# plt.subplots_adjust(left=0.2)
#
# lines = [l0, l1, l2]
#
# # Make checkbuttons with all plotted lines with correct visibility
# rax = plt.axes([0.05, 0.4, 0.1, 0.15])
# # labels = [str(line.get_label()) for line in lines]
# # visibility = [line.get_visible() for line in lines]
# labels = [key for key in checkboxes.keys()]
# visibility = [val for val in checkboxes.values()]
# check = CheckButtons(rax, labels, visibility)
#
#
# def func(label):
#     print(label)
#     # index = labels.index(label)
#     # lines[index].set_visible(not lines[index].get_visible())
#     # plt.draw()
#
# check.on_clicked(func)
# while True:
#     fig.canvas.flush_events()
# plt.show()
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.widgets import CheckButtons
# plt.ion()
# t = np.arange(0.0, 2.0, 0.01)
# s0 = np.sin(2*np.pi*t)
# s1 = np.sin(4*np.pi*t)
# s2 = np.sin(6*np.pi*t)
#
# fig, ax = plt.subplots()
# l0, = ax.plot(t, s0, visible=False, lw=2, color='k', label='2 Hz')
# l1, = ax.plot(t, s1, lw=2, color='r', label='4 Hz')
# l2, = ax.plot(t, s2, lw=2, color='g', label='6 Hz')
# plt.subplots_adjust(left=0.2)
#
# lines = [l0, l1, l2]
#
# # Make checkbuttons with all plotted lines with correct visibility
# rax = plt.axes([0.05, 0.4, 0.1, 0.15])
# labels = [str(line.get_label()) for line in lines]
# visibility = [line.get_visible() for line in lines]
# check = CheckButtons(rax, labels, visibility)
#
#
# def func(label):
#     index = labels.index(label)
#     lines[index].set_visible(not lines[index].get_visible())
#     plt.draw()
#
# check.on_clicked(func)
# while True:
#     fig.canvas.flush_events()
# # plt.show()