from .evolution.evolution_sampler import EvolutionSampler


def build_sampler(cfg, tester, net_cfg, cfg_stg_sample, **kwargs):
    kwargs = cfg.get('kwargs', {})
    # Only evolution sample is implemented
    return EvolutionSampler(tester=tester, net_cfg=net_cfg, cfg_stg_sample=cfg_stg_sample, cfg_sampler=cfg, **kwargs)
