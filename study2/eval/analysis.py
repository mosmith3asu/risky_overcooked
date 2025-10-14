from risky_overcooked_webserver.server.data_logging import DataViewer


class StatTests:
    def __init__(self):
        pass

    def risk_correlation(self):
        """Is there correlation between RPS-RisksTaken-Priming"""
        pass

    def t_test(self):
        pass

def main():
    fname = "cond_0/2025-09-02_16-12-04__PID123__cond0.pkl"
    viewer = DataViewer(fname,data_path='study2/human_data/data/')
    viewer.summary()


if __name__ == '__main__':
    main()