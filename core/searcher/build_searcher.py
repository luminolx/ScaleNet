from .uniform_searcher import UniformSearcher


def build_searcher(searcher_type, cfg_search_searcher, **kwargs):
    # Only uniform search is implemented
    return UniformSearcher(cfg_search_searcher, **kwargs)
