import logging
        
def create_logger(name, log_file, level=logging.INFO):
    """create logger for training"""
    logger = logging.getLogger(name)
    formatter = logging.Formatter('[%(asctime)s][%(filename)15s]'
                                  '[line:%(lineno)4d][%(levelname)8s]%(message)s')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.setLevel(level)
    logger.addHandler(fh)
    return logger
