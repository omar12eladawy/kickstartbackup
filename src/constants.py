import logging
import os

src_folder = os.path.dirname(os.path.abspath("__file__"))
destination_path = os.path.join(src_folder, "../data")
results_path= os.path.join(src_folder, "../results")

logging.basicConfig(level=logging.INFO, filename= os.path.join(results_path, "logs.txt"), force=True)
logger = logging.getLogger()