from importlib import import_module

from lib.msdata.dataloader import MSDataLoader
from torch.utils.data.dataloader import default_collate
 

class MSData:
    def __init__(self, args):
        self.loader_train = None
            
        module_train = import_module('msdata.dataset') # 'data.div2k'
        trainset = getattr(module_train, "ToyData")(args) # call data.div2.DIV2K
        self.loader_train = MSDataLoader(
            args,
            trainset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=not args.cpu
        )

        #module_test = import_module('data.' +  args.data_test.lower()) # 'data.div2k'
        #testset = getattr(module_test, args.data_test)(args, train=False) # from 0801 to 0810, specified by args.data_range
        # self.loader_test = MSDataLoader(
        #     args,
        #     testset,
        #     batch_size=args.batch_size, #1,
        #     shuffle=False,
        #     pin_memory=not args.cpu
        # )

