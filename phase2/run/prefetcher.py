from multiprocessing import Pool, cpu_count

import utils_run as utils
from image_processor import ImageProcessor
from data_source import DataSource

class PreFetcher:
    def __init__(self, datasource, batch_size=1, ahead=12):

        n_procs = cpu_count()
        self.pool = Pool(processes=n_procs)

        self.datasource = datasource
        self.batch_size = batch_size

        self.ahead = ahead
        self.batch_queue = []

    def fetch_batch(self):
        try:
            self.refill_tasks()
            result = self.batch_queue.pop(0)
            return result.get(31536000)[0]
        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, terminating workers")
            self.pool.terminate()
            raise

    def refill_tasks(self):
        # It will call the sequencer to ask for a sequence
        # of batch_size jobs (indexes with categories)
        # Then it will call pool.map_async
        while len(self.batch_queue) < self.ahead:
            idx = self.datasource.next_idx()
            print idx
            d = (self.datasource, idx, self.batch_size)
            batch = self.pool.map_async(build_batch, [d])
            self.batch_queue.append(batch)

def build_batch(d):
    datasource, idx, batch_size = d
    image_processor = ImageProcessor(datasource)
    batch = image_processor.process(idx, n=batch_size)
    return batch


if __name__ == "__main__":
    project = "ade20k"
    config = utils.get_config(project)

    datasource = DataSource(config, random=True)
    idx = 0
    batch_size = 5
    d = (datasource, idx, batch_size)
    batch = build_batch(d)

    data, label = batch
    print data.shape, label.shape

