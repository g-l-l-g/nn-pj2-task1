from torch.optim import lr_scheduler


def get_lr_scheduler(optimizer, exp_config, num_epochs):
    scheduler_type = exp_config.get("lr_scheduler_type")
    scheduler_params = exp_config.get("lr_scheduler_params", {})

    if scheduler_type is None:
        print("未配置学习率调度器。")
        return None
    scheduler_type_lower = scheduler_type.lower()
    print(f"尝试初始化学习率调度器: {scheduler_type_lower} with params: {scheduler_params}")

    if scheduler_type_lower == "steplr":
        step_size = scheduler_params.get("step_size", 30)
        gamma = scheduler_params.get("gamma", 0.1)
        print(f"  StepLR: step_size={step_size}, gamma={gamma}")
        return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type_lower == "cosineannealinglr":
        T_max = scheduler_params.get("T_max", num_epochs)
        eta_min = scheduler_params.get("eta_min", 0)
        print(f"  CosineAnnealingLR: T_max={T_max}, eta_min={eta_min}")
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    elif scheduler_type_lower == "multisteplr":
        milestones = scheduler_params.get("milestones")
        gamma = scheduler_params.get("gamma", 0.1)
        if milestones is None:
            print(f"警告: MultiStepLR 配置缺少 'milestones' 参数。将不使用调度器。")
            return None
        if not isinstance(milestones, list) or not all(isinstance(m, int) for m in milestones):
            print(f"警告: MultiStepLR 的 'milestones' 参数必须是一个整数列表。当前值: {milestones}。将不使用调度器。")
            return None
        print(f"  MultiStepLR: milestones={milestones}, gamma={gamma}")
        return lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    # --- 结束新增 ---
    elif scheduler_type_lower == "reducelronplateau":
        factor = scheduler_params.get("factor", 0.1)
        patience = scheduler_params.get("patience", 10)
        verbose = scheduler_params.get("verbose", True)
        print(f"  ReduceLROnPlateau: factor={factor}, patience={patience}")
        return lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, verbose=verbose)

    else:
        print(f"警告: 不支持的学习率调度器类型: {scheduler_type}。将不使用调度器。")
        return None