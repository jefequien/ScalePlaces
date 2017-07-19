from multiprocessing import Pool, cpu_count

import utils_run as utils
from image_processor import ImageProcessor
from data_source import DataSource

class PreFetcher:
    def __init__(self, image_processor, mode='train', batch_size=1, ahead=4):

        cpus = cpu_count()
        self.pool = Pool(processes=min(ahead, cpus))

        self.image_processor = image_processor
        self.batch_size = batch_size
        self.mode = mode

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
        while len(self.batch_queue) < self.ahead:
            im = self.image_processor.datasource.next_im()
            d = (self.image_processor, im, self.batch_size)

            if self.mode == 'train':
                batch = self.pool.map_async(build_train, [d])
                self.batch_queue.append(batch)
            elif self.mode == 'test':
                batch = self.pool.map_async(build_test, [d])
                self.batch_queue.append(batch)

def build_train(d):
    image_processor, im, batch_size = d
    batch = image_processor.build_data_and_label(im, batch_size=batch_size)
    return batch
def build_test(d):
    image_processor, im, batch_size = d
    batch = image_processor.build_data(im, batch_size=batch_size)
    return batch


