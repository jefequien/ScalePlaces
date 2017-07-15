
from multiprocessing import Pool, cpu_count


class DataLoader:
    def __init__(self, datasource, batch_size=1, ahead=12):
        n_procs = cpu_count()
        self.pool = Pool(processes=n_procs, initializer=setup_sigint)

        self.datasource = datasource
        self.batch_size = batch_size
        self.ahead = ahead

        self.batch_queue = []

    def fetch_batch(self):
        try:
            self.refill_tasks()
            return self.result_queue.pop(0)
        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, terminating workers")
            self.pool.terminate()
            raise

    def build_batch

    def refill_tasks(self):
        # It will call the sequencer to ask for a sequence
        # of batch_size jobs (indexes with categories)
        # Then it will call pool.map_async
        while len(self.result_queue) < self.ahead:
            data = [self.next_job() for _ in range(self.batch_size)]
            batch = self.pool.map_async(prefetch_worker, data)
            self.batch_queue.append(self.pool.map_async())
