def should_visualize(visualize_cfg, batch_idx):
    if visualize_cfg is None:
        return False
    
    interval = visualize_cfg.get("interval", 100)
    max_visualizations = visualize_cfg.get("max_visualizations", 10)
    if (batch_idx+1) % interval == 0 and (batch_idx // interval) < max_visualizations:
        return True
    
    return False