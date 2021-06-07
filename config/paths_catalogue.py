import os
PROJECT_PATH = os.path.abspath('..')

'''
Here provide paths to datasets you will use
Note that they always can be overwritten in config
I also like to put symlink to dataset directory in datasets dir in project
'''


class DatasetsCatalog(object):
    # DATASETS = {
    #     "DatasetName": {
    #         "path": os.path.join(PROJECT_PATH, 'datasets/datasetName'),
    #     }
    # }

    @staticmethod
    def get(name, **kwargs):
        # if "DatasetName" in name:
        #     path = DatasetsCatalog.DATASETS["DatasetName"]["path"]
        #     return path

        raise RuntimeError("Data not available: {}".format(name))
