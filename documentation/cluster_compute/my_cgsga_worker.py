# my_cgsga_worker.py
import pandas as pd
import numpy as np
from pyrates.utility.grid_search import ClusterWorkerTemplate


class MyWorker(ClusterWorkerTemplate):
    def worker_postprocessing(self, **worker_kwargs):

        # Pre-define index if only a single value is stored in each column
        self.processed_results = pd.DataFrame(index=["mean"], columns=self.results.columns)

        for idx, data in self.results.iteritems():
            self.processed_results.loc['mean', idx] = np.mean(data)


if __name__ == "__main__":
    cgs_worker = MyWorker()
    # cgs_worker.worker_test()
    cgs_worker.worker_init()
