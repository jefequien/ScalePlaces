
import utils_run as utils
from data_source import DataSource
from prefetcher import PreFetcher


project = "ade20k"
config = utils.get_config(project)

data_source = DataSource(config, random=True)
prefetcher = PreFetcher(data_source, batch_size=5, ahead=12)

data, label = prefetcher.fetch_batch()
print data.shape, label.shape