import torch
import matplotlib.pyplot as plt
import yaml
from models import DiT, SimpleMLPAdaLN

def build_model(model_type:str, config:dict):
    # 根据配置创建 DiT 模型
    assert config[model_type], f"Model type {model_type} not found in config"
    model_config = config[model_type]
    if model_type == "DiT":
        model = DiT(
            depth=model_config['depth'],
            hidden_size=model_config['hidden_size'],
            patch_size=model_config['patch_size'],
            num_heads=model_config['num_heads'],
            in_channels=model_config['in_channels'],
            input_size=model_config['input_size'],
            num_classes=model_config['num_classes']
        )
    elif model_type == "SimpleMLPAdaLN":
        model = SimpleMLPAdaLN(
            in_channels=model_config['in_channels'],
            model_channels=model_config['model_channels'],
            out_channels=model_config['out_channels'],
            z_channels=model_config['z_channels'],
            num_res_blocks=model_config['num_res_blocks']
        )
    return model   

def sample_and_plot(model, 
                    diffusion, 
                    output_dir, 
                    c = 0,
                    guidance_scale = [1.0], # if scale range if not given, use 1.0 (w/o guidance)
                    num_sample:int = 5000,
                    num_class:int = 2, 
                    gmodel = None,
                    ori_dist = None, 
                    comment:str="",
                    plot_color = "blue",
                    device="cuda:0",
                    ):
    
    if type(guidance_scale) == float:
        guidance_scale = [guidance_scale]
    
    use_ag = True if gmodel else False # if gmodel is given, use autoguidance

    for gs in guidance_scale:    
        class_labels = [c] * num_sample
        n = len(class_labels)
        y = torch.tensor(class_labels, device=device)
        y_null = torch.tensor([num_class] * n, device=device)
        
        if type(model) == DiT:
            z = torch.randn(n, 2, 1, 1, device=device)
        elif type(model) == SimpleMLPAdaLN:
            z = torch.randn(n, 2, device=device)
            y = y.unsqueeze(1).float()
            y_null = y_null.unsqueeze(1).float()
        else:
            raise NotImplementedError
        
        if use_ag:
            model_kwargs = dict(y=y, y_null=y_null, gs=gs, model_guide=gmodel)
            forward_func = model.forward_with_ag
        else:
            z = torch.cat([z, z], 0)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=gs)
            forward_func = model.forward_with_cfg

        # Sample images:
        samples = diffusion.p_sample_loop(
            forward_func, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        samples_cpu = samples.to('cpu')

        if type(model) == DiT:
            x = samples_cpu[:, 0, 0, 0].numpy()
            y = samples_cpu[:, 1, 0, 0].numpy()
        elif type(model) == SimpleMLPAdaLN:
            x = samples_cpu[:, 0].numpy()
            y = samples_cpu[:, 1].numpy()
        else:
            raise NotImplementedError
        
        plt.figure(figsize=(8, 8))
        if ori_dist is not None:
            plt.scatter(ori_dist[:,0], ori_dist[:,1], s=15, alpha=0.7, edgecolors='none', color="grey")
        plt.scatter(x, y, s=5, alpha=0.5, color=plot_color)
        plt.title(f'Scatter Plot (ag scale {gs})')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlim(-4, 4)
        plt.ylim(-4, 4)
        plt.grid(True)
        guid_type = "ag" if use_ag else "cfg"
        plt.savefig(f"{output_dir}/{comment}-{guid_type}_scale_{gs}.png")