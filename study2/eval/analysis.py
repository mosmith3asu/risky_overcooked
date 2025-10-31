import numpy as np
import matplotlib.pyplot as plt

from study2.eval.process import DataPoint,TorchPolicy
from study2.static import *

COLOR = {'oracle': tuple([50 / 255 for _ in range(3)]),
         'rs-tom':   (128 / 255, 0, 0),
         'rational': (255 / 255, 154 / 255, 0)
         # 'seeking': (128 / 255, 0, 0),
         # 'averse':  (255 / 255, 154 / 255, 0)
         }

LINE_STYLE = {'cond0':'-',
              'cond1':'-',
              'all':'-',
              'lbf':':'
              }
class StatTests:
    def __init__(self):
        pass

    def risk_correlation(self):
        """Is there correlation between RPS-RisksTaken-Priming"""
        pass

    def t_test(self):
        pass


# class PlotterBase:
#     def __init__(self,dp_list):
#         self.dp_list = dp_list


class TrustPlot:
    """
    Plots various trust-related metrics from a list of DataPoints with :

    Main Data: ---------------------
    - rewards (list)
    - predictability (list)
    - trust_scores: (list)
    - delta_trusts (list)
    - risk_perceptions (list)
    - C_ACTs (list)
    - H_IDLEs (list)
    - R_IDLEs (list)
    - nH_risks (list)
    - nR_risks (list)
    - relative_risk_perceptions (float)
    - relative_trust_scores (float)

     Mediating Variables: ------------
    - age (float)
    - sex (str)
    - RTP_score (float)
    - icond (int)
    - is_valid (boolean)
    - priming_labels (list)
    - priming_scores (list)

    TBD:
    - validity_score
    - survey_responses

    """
    def __init__(self, dp_list, scope):

        self.dp_list = dp_list
        self.scope = scope

    def radar_plot(self, ax, title=''):
        pass

    def _draw_lbf(self, ax,x, data, **kwargs):
        """Draws line of best fit onto given axis"""
        # x = np.arange(data.shape[1])
        y = np.mean(data, axis=0)
        coeffs = np.polyfit(x, y, 1)
        poly = np.poly1d(coeffs)
        ax.plot(x, poly(x), **kwargs)

    def timeseries_plot(self, ax, metric,
                        lbf=True, std=True,
                        offset=None,
                        title='',std_alpha=0.05):
        rs_tom, rational = [], []

        for dp in self.dp_list:
            # rs_tom.append(dp.trust_scores['rs-tom'])
            # rational.append(dp.trust_scores['rational'])
            rs_tom.append(  eval(f"dp.{metric}['rs-tom']" ))
            rational.append(eval(f"dp.{metric}['rational']"))


        rs_tom_style   = {'color': COLOR['rs-tom']  , 'linestyle': LINE_STYLE[self.scope]}
        rational_style = {'color': COLOR['rational'], 'linestyle': LINE_STYLE[self.scope]}

        rs_tom = np.array(rs_tom)
        rational = np.array(rational)


        T = rs_tom.shape[1]
        x_rs_tom = np.arange(T)
        x_rational = np.arange(T)
        if offset is not None:
            x_rs_tom += (T - 1 if offset == 'rs-tom' else 0)
            x_rational += (T - 1 if offset == 'rational' else 0)

        ax.plot(x_rs_tom, np.mean(rs_tom, axis=0),
                label=f'rs-tom   (M={np.mean(rs_tom):.2f})',
                **rs_tom_style
                )
        ax.plot(x_rational, np.mean(rational, axis=0),
                label=f'rational (M={np.mean(rational):.2f})',
                **rational_style
                )

        if std:
            ax.fill_between(x_rs_tom, np.mean(rs_tom, axis=0) - np.std(rs_tom, axis=0),
                            np.mean(rs_tom, axis=0) + np.std(rs_tom, axis=0), alpha=std_alpha,
                            **rs_tom_style
                            )
            ax.fill_between(x_rational, np.mean(rational, axis=0) - np.std(rational, axis=0),
                            np.mean(rational, axis=0) + np.std(rational, axis=0),alpha=std_alpha,
                            **rational_style
                            )

        if lbf:
            rs_tom_style['linestyle'] = LINE_STYLE['lbf']
            rational_style['linestyle'] = LINE_STYLE['lbf']
            self._draw_lbf(ax, x_rs_tom, rs_tom, **rs_tom_style)
            self._draw_lbf(ax, x_rational, rational, **rational_style)
        ax.set_title(title)
        ax.legend()

    def delta_plot(self, ax, metric,title=''):
        rs_tom, rational = [], []
        for dp in self.dp_list:
            # rs_tom.append(dp.delta_trusts['rs-tom'])
            # rational.append(dp.delta_trusts['rational'])
            rs_tom.append(eval(f"dp.{metric}['rs-tom']"))
            rational.append(eval(f"dp.{metric}['rational']"))
        rs_tom = np.array(rs_tom)
        rational = np.array(rational)
        rs_tom_style = {'color': COLOR['rs-tom']}
        rational_style = {'color': COLOR['rational']}

        mean_rs_tom = np.mean(rs_tom)
        mean_rational = np.mean(rational)
        std_rs_tom = np.mean(np.std(rs_tom, axis=0))
        std_rational = np.mean(np.std(rational, axis=0))
        ax.bar(0, mean_rs_tom, yerr=std_rs_tom, label='rs-tom', alpha=0.7,**rs_tom_style)
        ax.bar(1, mean_rational, yerr=std_rational, label='rational', alpha=0.7,**rational_style)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['rs-tom', 'rational'])
        #
        ax.set_title(title)
        # ax.legend()



def main():
    COND0_FNAMES = [
        "2025-10-16_22-37-54__PID68c08483061276d20b570579__cond0",
        "2025-10-28_23-12-28__PID58a0c507890ea500014c4e9b__cond0",
        "2025-10-27_18-53-52__PID67062295123561f8241f65fc__cond0"
    ]
    COND1_FNAMES = [
        "2025-10-28_22-46-41__PID5dce29700ad506063969a4a5__cond1",
        "2025-10-16_22-32-23__PID672135003f5e272c889620ea__cond1"
    ]
    # fname = "2025-10-16_22-37-54__PID68c08483061276d20b570579__cond0"
    # fpath = PROCESSED_COND0_DIR + r"\\"+ fname
    # dp = DataPoint.load_processed(fpath)
    dps_cond1 = []
    dps_cond0 = []
    dps_all = []
    for fname in COND0_FNAMES:
        fpath = PROCESSED_COND0_DIR + r"\\"+ fname
        dp = DataPoint.load_processed(fpath)
        dps_cond0.append(dp)

    for fname in COND1_FNAMES:
        fpath = PROCESSED_COND1_DIR + r"\\"+ fname
        dp = DataPoint.load_processed(fpath)
        dps_cond1.append(dp)

    dps_all += dps_cond0 + dps_cond1
    nrow, ncol = 3,3

    fig, axs = plt.subplots(nrow, ncol, constrained_layout=True, figsize=(ncol*5, nrow*4))

    plt.ioff()
###############################
    trust_plot = TrustPlot(dps_cond1,scope='cond1')
    offset = 'rs-tom'
    cond_label = 'Cond1'
    # fig.suptitle(f'Metrics - {cond_label}: Rational -> RS-ToM', fontsize=16)

    trust_plot.timeseries_plot(axs[0, 0], 'trust_scores', title=f'({cond_label}) Trust Over Time', offset=offset)
    trust_plot.delta_plot(axs[0, 1], 'delta_trusts', title=f'({cond_label}) Delta Trust')


    trust_plot.timeseries_plot(axs[1, 0], 'C_ACTs', title=f'({cond_label}) C_ACTS', offset=offset)
    trust_plot.timeseries_plot(axs[1, 1], 'H_IDLEs', title=f'({cond_label}) H_IDLEs', offset=offset)
    trust_plot.timeseries_plot(axs[1, 2], 'R_IDLEs', title=f'({cond_label}) R_IDLEs', offset=offset)

    trust_plot.timeseries_plot(axs[2, 0], 'rewards', title=f'({cond_label}) Reward', offset=offset)
    trust_plot.timeseries_plot(axs[2, 1], 'predictability', title=f'({cond_label}) Predictability', offset=offset)
    trust_plot.delta_plot(axs[2, 2], 'risk_perceptions', title=f'({cond_label}) Risk Perception')

    ###############################
    trust_plot = TrustPlot(dps_cond0, scope='cond0')
    offset = 'rational'
    cond_label = 'Cond0'
    fig.suptitle(f'Metrics - {cond_label}: RS-ToM -> Rational', fontsize=16)

    trust_plot.timeseries_plot(axs[0, 0], 'trust_scores', title=f'({cond_label}) Trust Over Time', offset=offset)
    trust_plot.delta_plot(     axs[0, 1], 'delta_trusts', title=f'({cond_label}) Delta Trust')
    trust_plot.delta_plot(     axs[0, 2], 'risk_perceptions', title=f'({cond_label}) Risk Perception')

    trust_plot.timeseries_plot(axs[1, 0], 'C_ACTs', title=f'({cond_label}) C_ACTS', offset=offset)
    trust_plot.timeseries_plot(axs[1, 1], 'H_IDLEs', title=f'({cond_label}) H_IDLEs', offset=offset)
    trust_plot.timeseries_plot(axs[1, 2], 'R_IDLEs', title=f'({cond_label}) R_IDLEs', offset=offset)

    trust_plot.timeseries_plot(axs[2, 0], 'rewards', title=f'({cond_label}) Reward', offset=offset)
    trust_plot.timeseries_plot(axs[2, 1], 'predictability', title=f'({cond_label}) Predictability', offset=offset)

    ###############################

    # trust_plot = TrustPlot(dps_all, scope='cond0')
    # offset = None
    # cond_label = 'All'
    # fig.suptitle(f'Metrics - {cond_label}', fontsize=16)
    #
    # trust_plot.timeseries_plot(axs[0, 0], 'trust_scores', title=f'({cond_label}) Trust Over Time', offset=offset)
    # trust_plot.delta_plot(axs[0, 1], 'delta_trusts', title=f'({cond_label}) Delta Trust')
    # trust_plot.delta_plot(axs[0, 2], 'risk_perceptions', title=f'({cond_label}) Risk Perception')
    #
    # trust_plot.timeseries_plot(axs[1, 0], 'C_ACTs', title=f'({cond_label}) C_ACTS', offset=offset)
    # trust_plot.timeseries_plot(axs[1, 1], 'H_IDLEs', title=f'({cond_label}) H_IDLEs', offset=offset)
    # trust_plot.timeseries_plot(axs[1, 2], 'R_IDLEs', title=f'({cond_label}) R_IDLEs', offset=offset)
    #
    # trust_plot.timeseries_plot(axs[2, 0], 'rewards', title=f'({cond_label}) Reward', offset=offset)
    # trust_plot.timeseries_plot(axs[2, 1], 'predictability', title=f'({cond_label}) Predictability', offset=offset)


    plt.show()



if __name__ == '__main__':
    main()