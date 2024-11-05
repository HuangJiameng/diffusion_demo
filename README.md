# diffusion_demo

这是一个简单的扩散模型示例。请将 `image_path` 设为一个黑白图片的路径，该图片将模拟一个二维数据分布：图中的黑色像素为样本点，左半边标签为0，右半边标签为1。你可以用 `model_type` 参数选定要训练的模型种类，目前支持 "DiT" 与 "SimpleMLPAdaLN"。你可以在 config 文件夹下添加 yaml 格式的参数文件设置模型超参数，并用 `config_path` 指定 yaml 文件路径（默认读取 config 文件夹下的 `default.yml`）。

训练命令样例：`python train.py --output_dir './dit_outputs' --model_type 'DiT'`。

更多参数详见 `train.py` 的 `parse_args` 函数。

This is a simple diffusion model toy example. Please set the `image_path` to the path of a black and white image, which will simulate a two-dimensional data distribution: the black pixels in the image are the sample points, with a label of 0 for the left half and a label of 1 for the right half. You can use the `model_type` parameter to select the type of model to train, currently supporting "DiT" and "SimpleMLPAdaLN". You can add a yaml-formatted parameter file under the config folder to set model hyperparameters, and specify the yaml file path with `config_path` (default reads the `default.yml` in the config folder).

An example training command is: `python train.py --output_dir './dit_outputs' --model_type 'DiT'`. 

For more parameters, see the `parse_args` function in `train.py`.
