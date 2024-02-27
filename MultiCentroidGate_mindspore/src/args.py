import argparse
import yaml
import utils


class IncrementalConfig(utils.EasyConfig):
    def __init__(self):
        super().__init__()
        self.increments = [] # number to expand classifier.
        self.unique_increments = [] # for overlap situation
        self.idx_task = -1
        self.nb_seen_unique_classes = 0
        self.nb_seen_classes = 0
        self.nb_task_classes = 0
        self.nb_task_unique_classes = 0

        self.gpu = -1

    @property
    def nb_tasks(self): 
        return len(self.increments)
    
    @property
    def nb_prev_classes(self):
        return self.nb_seen_classes - self.nb_task_classes
 
    def new_task(self, nb_task_classes, nb_task_unique_classes=-1):
        if nb_task_unique_classes == -1:
            nb_task_unique_classes = nb_task_classes

        self.idx_task += 1
        self.nb_task_classes = nb_task_classes
        self.nb_task_unique_classes = nb_task_unique_classes
        
        self.increments.append(nb_task_classes)
        self.unique_increments.append(nb_task_unique_classes)

        self.nb_seen_classes += nb_task_classes
        self.nb_seen_unique_classes += nb_task_unique_classes 

 
def init_cfg():
    cfg = IncrementalConfig()
    parser = argparse.ArgumentParser() 
    parser.add_argument("--config-files", default=["conf/config_ok.yaml"], nargs='+') 
    a, _ = parser.parse_known_args()
    yaml_config = {}
    for cf in a.config_files:
        with open(cf, "r") as f:
            yaml_config= utils.deep_update(yaml_config, yaml.safe_load(f))
            # yaml_config.update(yaml.safe_load(f)) 

    parser = argparse.ArgumentParser() 
    utils.add_argument_using_dict(parser, yaml_config)
    parser.add_argument("--config-files", default=["conf/config_ok.yaml"], nargs='+') 
    console_config, unknown = parser.parse_known_args() 
    if len(unknown) != 0:
        print(unknown)
        raise Exception()

    cfg.__dict__.update(yaml_config)
    utils.merge_args(console_config.__dict__, cfg.__dict__)
    
    if cfg['debug']:
        cfg.warmup_epochs = 1
        cfg.epochs = 1
        cfg.ft.epochs = 1 
        cfg.save_model = False  
    return cfg
