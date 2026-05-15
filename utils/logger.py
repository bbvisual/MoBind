import logging

def setup_logging(log_file=None, level=logging.INFO):
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d,%H:%M:%S'
    )

    # Avoid adding duplicate handlers if this function is called multiple times
    if logging.getLogger().hasHandlers():
        logging.getLogger().handlers.clear()

    # Set root logger level
    logging.root.setLevel(level)

    # Console handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logging.root.addHandler(stream_handler)

    file_handler = logging.FileHandler(filename=log_file)
    file_handler.setFormatter(formatter)
    logging.root.addHandler(file_handler)
