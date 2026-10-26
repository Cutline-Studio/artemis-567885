"""Method 11: grow the model while it learns (function-preserving depth and FFN-width expansion).

Uses ArtemisDense.add_layer() (new block's output projections start at zero -> identity) and
widen_ffn() (new FFN units' down-projection starts at zero), so the loss right after growth is
unchanged and training continues from there. d_model widening is not function-preserving with
tied RMSNorm stats and is not implemented.
"""
import torch


def shrink_for_growth(cfg_dict: dict, start_layers: int) -> dict:
    return {**cfg_dict, "n_layers": max(1, min(start_layers, cfg_dict["n_layers"]))}


@torch.no_grad()
def grow(model, add_layers: int = 1, new_d_ff: int = 0, generator=None) -> dict:
    before = (model.num_layers, model.cfg.d_ff, model.num_params())
    for _ in range(add_layers):
        model.add_layer()
    if new_d_ff and new_d_ff > model.cfg.d_ff:
        model.widen_ffn(new_d_ff, generator=generator)
    return {"layers": [before[0], model.num_layers], "d_ff": [before[1], model.cfg.d_ff],
            "params": [before[2], model.num_params()]}


def rebuild_optimizer(opt, model):
    """New params need optimizer state; keep the old state for the params that survived."""
    old = {id(p): opt.state[p] for g in opt.param_groups for p in g["params"] if p in opt.state}
    import inspect
    ok = set(inspect.signature(type(opt).__init__).parameters)
    new = type(opt)([p for p in model.parameters() if p.requires_grad],
                    **{k: v for k, v in opt.defaults.items() if k in ok})
    for p in model.parameters():
        if id(p) in old:
            new.state[p] = old[id(p)]
    return new
