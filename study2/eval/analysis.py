import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from study2.eval.process import DataPoint,TorchPolicy
from study2.eval.eval_utils import get_processed_fnames
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
        self.outlier_thresh = 2 #std deviations

    def radar_plot(self, ax, title=''):
        pass

    def _draw_lbf(self, ax,x, data, **kwargs):
        """Draws line of best fit onto given axis"""
        # x = np.arange(data.shape[1])
        y = np.mean(data, axis=0)
        coeffs = np.polyfit(x, y, 1)
        poly = np.poly1d(coeffs)
        ax.plot(x, poly(x), **kwargs)

    def remove_outliers(
            self,
            X,
            mode: str = "by_time",  # "by_time" or "by_series"
            thresh: float = 3.5,  # robustness threshold (≈ 3–4 is common)
            fix: str = "median",  # "median", "nan", or "interpolate"
            eps: float = 1e-12  # numerical floor for MAD
    ):
        """
        Remove/repair outliers in an N x T array using robust median + MAD.

        Parameters
        ----------
        X : np.ndarray
            Array of shape (N, T).
        mode : {"by_time", "by_series"}
            - "by_time": detect outliers at each timestep across samples (axis=0).
            - "by_series": detect outliers within each series across time (axis=1).
        thresh : float
            Outlier threshold in scaled MAD units (3–4 is typical).
        fix : {"median", "nan", "interpolate"}
            How to repair flagged points:
              - "median": replace with the robust center (median per time or per series).
              - "nan": set to NaN.
              - "interpolate": linear interpolation along time for each row.
                (Only meaningful if T >= 2; works across interior NaN runs.)
        eps : float
            Small constant to avoid division by zero when MAD==0.

        Returns
        -------
        X_fixed : np.ndarray
            Cleaned array (same shape as X).
        mask : np.ndarray (bool)
            Boolean mask of shape (N, T) where True indicates an outlier.
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (N, T).")
        N, T = X.shape

        if mode not in {"by_time", "by_series"}:
            raise ValueError("mode must be 'by_time' or 'by_series'.")
        if fix not in {"median", "nan", "interpolate"}:
            raise ValueError("fix must be 'median', 'nan', or 'interpolate'.")

        X_fixed = X.copy()

        # Robust center and scale
        if mode == "by_time":
            # center/scale per timestep (axis=0)
            center = np.median(X, axis=0, keepdims=True)  # (1, T)
            mad = np.median(np.abs(X - center), axis=0, keepdims=True)  # (1, T)
            scale = 1.4826 * np.maximum(mad, eps)
        else:  # "by_series"
            # center/scale per series (axis=1)
            center = np.median(X, axis=1, keepdims=True)  # (N, 1)
            mad = np.median(np.abs(X - center), axis=1, keepdims=True)  # (N, 1)
            scale = 1.4826 * np.maximum(mad, eps)

        z = (X - center) / scale
        mask = np.abs(z) > thresh

        # Repair strategy
        if fix == "median":
            X_fixed[mask] = center[mask]  # broadcast matches the computed center
        elif fix == "nan":
            X_fixed[mask] = np.nan
        else:  # "interpolate"
            # Interpolate along time for each series (row-wise), leaving non-flagged values untouched.
            # We set flagged values to NaN, then fill NaNs via 1D linear interpolation over time.
            X_nan = X_fixed.astype(float).copy()
            X_nan[mask] = np.nan

            # Build time index once
            t_idx = np.arange(T)
            for i in range(N):
                row = X_nan[i]
                good = ~np.isnan(row)
                if good.sum() == 0:
                    # All points are NaN: fall back to robust center for that row
                    if mode == "by_time":
                        # Use per-timestep centers
                        X_fixed[i] = center[0]
                    else:
                        # Use per-series center
                        X_fixed[i] = center[i, 0]
                    continue
                if good.sum() == 1:
                    # Only one good point: constant fill
                    X_fixed[i] = np.where(np.isnan(row), row[good][0], row)
                    continue

                # Linear interpolate across NaNs
                interp_vals = np.interp(t_idx, t_idx[good], row[good])
                X_fixed[i] = interp_vals

        return X_fixed, mask

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
        # for mode in ["by_time", "by_series"]:
        # for mode in ["by_series"]:
        #     rs_tom, rs_tom_mask = self.remove_outliers(rs_tom, mode=mode, thresh=self.outlier_thresh , fix="interpolate")
        #     rational, rational_mask = self.remove_outliers(rational, mode=mode, thresh=self.outlier_thresh , fix="interpolate")

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

        # for mode in ["by_time"]:
        #     rs_tom, rs_tom_mask = self.remove_outliers(rs_tom, mode=mode, thresh=self.outlier_thresh , fix="interpolate")
        #     rational, rational_mask = self.remove_outliers(rational, mode=mode, thresh=self.outlier_thresh , fix="interpolate")


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


def plot_trust_radar(data_list):
    """Plots radar chart of trust survey responses for given data points"""
    tom_response_dict = {}
    rat_response_dict = {}


    for dp in data_list:
        for partner in ['rs-tom', 'rational']:
            for isurvey, survey in enumerate(dp.trust_responses[partner]):
                for qkey, response in survey.items():
                    if partner == 'rs-tom':
                        if qkey not in tom_response_dict:
                            tom_response_dict[qkey] = []
                        tom_response_dict[qkey].append(response)
                    else:
                        if qkey not in rat_response_dict:
                            rat_response_dict[qkey] = []
                        rat_response_dict[qkey].append(response)

    # make radar plot
    labels = list(tom_response_dict.keys())
    tom_means = [np.mean(tom_response_dict[label]) for label in labels]
    rat_means = [np.mean(rat_response_dict[label]) for label in labels]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    tom_means += tom_means[:1]
    rat_means += rat_means[:1]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, tom_means, color=COLOR['rs-tom'], alpha=0.25, label='RS-ToM')
    ax.fill(angles, rat_means, color=COLOR['rational'], alpha=0.25, label='Rational')
    ax.plot(angles, tom_means, color=COLOR['rs-tom'], linewidth=2)
    ax.plot(angles, rat_means, color=COLOR['rational'], linewidth=2)
    ax.set_yticklabels([])
    ax.set_ylim(0,10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title('Trust Survey Responses', size=20, y=1.05)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.show()


def preview_data(data):
    data_df = data[0].to_pandas_flat()
    for dp in data[1:]:
        new_df = dp.to_pandas_flat()
        data_df = pd.concat([data_df, new_df], ignore_index=True)

    cols = ['rs-tom_mean_dtrust','rational_mean_dtrust']
    df_supports = data_df[cols[0]] > data_df[cols[1]]
    df_diff = data_df[cols[0]] - data_df[cols[1]]

    data_df['rs-tom > rational'] = np.where(df_supports, 1, 0)
    data_df['tom-rat'] = df_diff

    print(data_df[['ID', 'condition', 'rs-tom > rational','tom-rat'] ])
    # print(data_df[['ID','condition','rs-tom > rational'] + cols])
    perc_supports = df_supports.sum() / len(df_supports) * 100
    print(f"%Support: {perc_supports:.2f}%")

def main():
    fname_dict = get_processed_fnames(full_path=False)
    COND1_FNAMES = fname_dict['cond1']
    COND0_FNAMES = fname_dict['cond0']

    dps_cond1 = []
    dps_cond0 = []
    dps_all = []
    for fname in COND0_FNAMES:
        fpath = PROCESSED_COND0_DIR + "\\"+ fname
        dp = DataPoint.load_processed(fpath)
        dps_cond0.append(dp)

    for fname in COND1_FNAMES:
        fpath = PROCESSED_COND1_DIR + "\\"+ fname
        dp = DataPoint.load_processed(fpath)
        dps_cond1.append(dp)


    dps_all += dps_cond0 + dps_cond1
    plot_trust_radar(dps_cond0)
    preview_data(dps_all)
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
    trust_plot.delta_plot(axs[0, 2], 'trust_slopes', title=f'({cond_label}) Slope Trust')


    trust_plot.timeseries_plot(axs[1, 0], 'C_ACTs', title=f'({cond_label}) C_ACTS', offset=offset)
    trust_plot.timeseries_plot(axs[1, 1], 'H_IDLEs', title=f'({cond_label}) H_IDLEs', offset=offset)
    trust_plot.timeseries_plot(axs[1, 2], 'R_IDLEs', title=f'({cond_label}) R_IDLEs', offset=offset)

    trust_plot.timeseries_plot(axs[2, 0], 'rewards', title=f'({cond_label}) Reward', offset=offset)
    trust_plot.timeseries_plot(axs[2, 1], 'predictability', title=f'({cond_label}) Predictability', offset=offset)
    # trust_plot.delta_plot(axs[2, 2], 'risk_perc_scores', title=f'({cond_label}) Risk Perception')

    ###############################
    # trust_plot = TrustPlot(dps_cond0, scope='cond0')
    # offset = 'rational'
    # cond_label = 'Cond0'
    # fig.suptitle(f'Metrics - {cond_label}: RS-ToM -> Rational', fontsize=16)
    #
    # trust_plot.timeseries_plot(axs[0, 0], 'trust_scores', title=f'({cond_label}) Trust Over Time', offset=offset)
    # trust_plot.delta_plot(axs[0, 1], 'delta_trusts', title=f'({cond_label}) Delta Trust')
    #
    #
    # trust_plot.timeseries_plot(axs[1, 0], 'C_ACTs', title=f'({cond_label}) C_ACTS', offset=offset)
    # trust_plot.timeseries_plot(axs[1, 1], 'H_IDLEs', title=f'({cond_label}) H_IDLEs', offset=offset)
    # trust_plot.timeseries_plot(axs[1, 2], 'R_IDLEs', title=f'({cond_label}) R_IDLEs', offset=offset)
    #
    # trust_plot.timeseries_plot(axs[2, 0], 'rewards', title=f'({cond_label}) Reward', offset=offset)
    # trust_plot.timeseries_plot(axs[2, 1], 'predictability', title=f'({cond_label}) Predictability', offset=offset)
    # # trust_plot.delta_plot(axs[2, 2], 'risk_perc_scores', title=f'({cond_label}) Risk Perception')

    ###############################

    # trust_plot = TrustPlot(dps_all, scope='all')
    # offset = None
    # cond_label = 'All'
    # fig.suptitle(f'Metrics - {cond_label}', fontsize=16)
    #
    # trust_plot.timeseries_plot(axs[0, 0], 'trust_scores', title=f'({cond_label}) Trust Over Time', offset=offset)
    # trust_plot.delta_plot(axs[0, 1], 'delta_trusts', title=f'({cond_label}) Delta Trust')
    # # trust_plot.delta_plot(axs[0, 2], 'risk_perc_scores', title=f'({cond_label}) Risk Perception')
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