
import utils_run as utils
from data_source import DataSource
from data_loader import DataLoader


project = "local"
config = utils.get_config(project)

data_source = DataSource(config, random=True)
data_loader = DataLoader(data_source, batch_size=5, ahead=12)

data, label = data_loader.fetch_batch()
print data.shape, label.shape