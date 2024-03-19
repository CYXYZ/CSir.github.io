---
layout:     post
title:      Diffuse Morph代码笔记
subtitle:   详解论文与代码对应关系
date:       2024-03-17
author:     CYXYZ
header-img: img/diff-pose.jpg
catalog: true
tags:
    - Diffusion model
    - 2D and 3D
---

# Diffuse Morph notebook

## Diffuse Morph论文主体思想

文章DiffuseMorph: Unsupervised Deformable
Image Registration Using Diffusion Model 提出了一种新的基于扩散模型的图像配准方法，称为DiffuseMorph。DiffuseMorph不仅可以通过反向扩散生成合成变形图像，还可以通过变形场进行图像配准。

我们这里主要研究文章的主要思路

### 文章概述

在本文中，利用扩散模型的特性，其中估计的潜在特征提供空间信息来生成图像，我们提出了一种新的无监督形变图像配准方法，称为DiffuseMorph，通过调整DDPM来生成变形场。

文章中，提出的模型由扩散网络和变形网络组成:**扩散网络学习运动和固定图像之间变形的条件分数函数**，**变形网络利用分数函数的潜在特征估计变形场并提供变形图像**。

整体是一个端到端的模型。

*这里加个小注释吧：端到端就相当于把网络比作了一个黑盒，黑盒中的内容没有什么具体的可解释性和人为参与，我们只关注输入输出和最终的模型效果。*

*端到端模型是一种能够直接从原始输入数据到最终输出结果完成整个任务的模型。换句话说，端到端模型不需要人为地对输入数据进行手动处理或者中间步骤的人工设计，而是通过学习从输入到输出的映射关系来完成整个任务。*

*在端到端模型中，通常包含了从原始数据到目标结果的所有处理步骤，这些步骤可能包括特征提取、特征转换、特征选择、模式识别、决策等等。而且，端到端模型通常是由深度学习模型构成的，例如卷积神经网络（CNN）、循环神经网络（RNN）、自动编码器（Autoencoder）等。*

<u>**其实如果可以的话这里的两个网络和配准中的两个网络也比较相似，只不过是把变形场估计换为了姿态估计的网络。可以把姿态估计的输出也看作是一种形变**</u>

文章中提到了一种过程，通过正向扩散一步传播运动图像，通过DDPM反向过程迭代细化。使得样本保留了原始的运动图像。 

### 去噪扩散概率模型

#### diffusion综述

正向扩散过程，使用马尔可夫链向数据$x_0$中加入噪声。对于$t\in[0,t]$，采样潜变量$x_t$被定义为高斯跃迁。

经过$T$轮，将输入图片$x_0$变为纯高斯噪声$x_T$，而模型负责将$x_T$复原为图像$x_0$。

##### 前向过程：

通过$T$次累计对其添加高斯噪声，得到$x_0...x_T$。如下面的公式所示，这里需要给定一系列高斯分布方差的超参数$\{\beta_t\in(0,1)\}_{t=1}^T$。前向过程由于每个时刻$t$只与$t-1$时刻相关，所以可以看作马尔可夫过程。

*马尔可夫过程不具备记忆特质，其条件概率仅与系统的当前状态相关，与其过去或未来都是独立，不相关的。*

令$p(x)$为图像$x$在生成过程中产生当前数据的可能性，若有$p(x)$，则可以得到给定一张图片的概率有多大。

利用下面的公式定义每一步增加的高斯分布：

$$q(x_t|x_{t-1})=N(x_t;\sqrt{1-\beta_t}x_{t-1},\beta_tI)$$

其中$I$是单位矩阵。

其中，$x_t$是第$t$步的随机变量。具体来说，$q(x_t|x_{t-1})$表示了给定$x_{t-1}$条件下，$x_t$的分布情况。在这个公式中，$x_t$ 是一个随机变量，它的取值是根据这个概率分布来生成的。可以理解为$q(x_t|x_{t-1})$是给定条件下的$x_{t-1}$的“可能性”。

具体来说，公式中的$\sqrt{1-\beta_t}x_{t-1}$可以看作是对$x_{t-1}$进行的微小扰动，而$\beta_tI$表示了这个扰动的具体大小。在生成$x_t$的过程中，通过在$x_{t-1}$上加入服从均值为$\sqrt{1-\beta_t}x_{t-1}$，方差为$\beta_tI$的高斯噪声来模拟数据的变化。

则，若给出$x_0$可以得到$x_t$

$$q(x_t|x_0)=\prod_{t=1}^t q(x_t|x_{t-1})$$

另外要指出的是，扩散过程往往是固定的，即采用一个预先定义好的**variance schedule**，比如DDPM就采用一个线性的**variance schedule**。

在根据两个方差不同的高斯分布相加等于一个新的高斯分布公式$N(0,\sigma_1^2I)$和$N(0,\sigma_2^2I)$相加等于一个新的高斯分布$N(0,(\sigma_1^2+\sigma_2^2)I)$，则可以得到

$q(x_t|x_0)=N(x_t;\sqrt\alpha_tx_0,(1-\alpha_t)I)$.

根据贝叶斯公式就可以得到

$p_\theta(x_{t-1}|x_t)=N(x_{t-1};\mu_\theta(x_t,t),\sum_\theta(x_t,t))$

##### 后向过程

这个反向过程的均值和方差由训练的网络$\mu_\theta(x_t,t)$和$\sum_\theta(x_t,t)$给出。文章中最后训练好的参数是$\epsilon_\theta$。

对于生成过程进行反向扩散，DDPM学习参数化高斯过程，这里的推导好麻烦，参考链接https://zhuanlan.zhihu.com/p/563661713

整个模型训练完之后，通过下面的随机生成步骤对数据进行采样：

$x_{t-1}=\mu_\theta(x_t,t)+\sigma_tz$，其中$\sigma^2_t=\frac{1-\alpha_{t-1}}{1-\alpha_t}\beta_t$，并且$z~N(0,I)$，$\alpha_t=\prod_{s=1}^t(1-\beta_s)$.

#### 网络结构

##### train部分

整个网络由两部分组成，一部份是扩散网络，用于轨迹条件分数函数。一部分是变形网络，用于分数函数输出配准段。

具体说，这里将需要配准的图像建模为源图像$m$，被配准后的源图像建模为固定的参考图像$f$，训练扩散网络$G_\theta$，在条件$c=(m,f)$下，学习两者之间的变形条件分数函数。

通过对目标的$x_t$进行采样，将固定的图像定义为目标，即$x_0=f$，为了使网络$G_\theta$意识到噪声的程度，给出了噪声随着网络的时间步长。

另一方面，变形网络$M_\phi$取扩散网络输出的条件分数函数的潜在特征和运动的源图像$m$，输出配准场$\phi$，利用空间变换层STL，得到变形图像$m(\phi)$，对运动图像$m$进行2D/3D的图像进行变形，采用双/三线性插值的变换函数。

##### test部分

整个网络被训练之后，推理时通过估计运动图像与固定图像对齐的变形场来提供图像配准。

#### 通过反向扩散生成合成图像

由于扩散网络学习了运动图像和固定图像之间的变形的条件分数函数，因此图像生成的任务从运动图像开始，而不是从纯高斯噪声开始

## 代码的主要结构

在main_2D代码中，diffusion被定义为model.py中的DDPM类。因此我们重点来解析这个类中的函数。

### DDPM类

继承自**Basemodel**类，来自文件model/base_model.py

```python
class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device(
            'cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.begin_step = 0
        self.begin_epoch = 0

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def get_current_losses(self):
        pass

    def print_network(self):
        pass

    def set_device(self, x):
        if isinstance(x, dict):
            for key, item in x.items():
                if item is not None and not isinstance(item, list):
                    x[key] = item.to(self.device)
        elif isinstance(x, list):
            for item in x:
                if item is not None:
                    item = item.to(self.device)
        else:
            x = x.to(self.device)
        return x

    def get_network_description(self, network):
        '''Get the string and total parameters of the network'''
        if isinstance(network, nn.DataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n
```

这个类实际只定义了两个基本函数`set_device` 函数使得在训练过程中方便地将数据移动到合适的设备上进行计算，而 `get_network_description` 函数则提供了网络结构信息和参数数量，方便网络的调试和分析。

1. `set_device`函数接收一个张量，并且将其移动到 `BaseModel` 类中初始化时指定的设备上。如果输入是字典，则遍历字典中的每个项目，将其值（假设不是列表）移动到指定设备上。如果输入是列表，则遍历列表中的每个张量，将其移动到指定设备上。如果输入是单个张量，则将其直接移动到指定设备上。移动操作通过调用 `to()` 方法完成。
2. `get_network_description`函数接收一个网络模型作为输入，并返回一个包含网络结构信息和总参数数量的元组 `(s, n)`。如果输入的网络模型是 `nn.DataParallel` 类型的，函数会先将其转换成普通的模型。然后，函数通过调用模型的 `__str__()` 方法获取网络的字符串表示形式（即结构信息），并通过 `numel()` 方法计算模型所有参数的总数量。最后，函数将结构信息字符串和总参数数量组成一个元组返回。

接下来看DDPM类

```python
class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)
        # define network and load pretrained models
        # 定义网络并加载预训练模型
        self.netG = self.set_device(networks.define_G(opt))

        self.schedule_phase = None
        self.centered = opt['datasets']['centered']

        # set loss and load resume state
        # 设置损失并加载恢复状态
        self.set_loss()
        self.set_new_noise_schedule(opt['model']['beta_schedule']['train'], schedule_phase='train')
        self.load_network()
        if self.opt['phase'] == 'train':
            self.netG.train()
            # find the parameters to optimize
            # 寻找要优化的参数
            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params = list(self.netG.parameters())

            self.optG = torch.optim.Adam(optim_params, lr=opt['train']["optimizer"]["lr"], betas=(0.5, 0.999))
            self.log_dict = OrderedDict()
        self.print_network(self.netG)

    def feed_data(self, data):
        self.data = self.set_device(data)

    def optimize_parameters(self):
        self.optG.zero_grad()
        score, loss = self.netG(self.data)
        self.score, self.out_M, self.flow = score
        # need to average in multi-gpu

        # l_tot = loss
        # 计算总损失并反向传播
        l_pix, l_sim, l_smt, l_tot = loss
        l_tot.backward()
        self.optG.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()
        self.log_dict['l_sim'] = l_sim.item()
        self.log_dict['l_smt'] = l_smt.item()
        self.log_dict['l_tot'] = l_tot.item()

    def test_generation(self, continuous=False):
        self.netG.eval()
        input = torch.cat([self.data['M'], self.data['F']], dim=1)
        if isinstance(self.netG, nn.DataParallel):
            self.MF = self.netG.module.generation(input, continuous)
        else:
            self.MF= self.netG.generation(input, continuous)
        self.netG.train()

    def test_registration(self, continuous=False):
        self.netG.eval()
        input = torch.cat([self.data['M'], self.data['F']], dim=1)
        nsample = self.data['nS']
        if isinstance(self.netG, nn.DataParallel):
            self.out_M, self.flow, self_contD, self.contF = self.netG.module.registration(input, nsample=nsample, continuous=continuous)
        else:
            self.out_M, self.flow, self.contD, self.contF = self.netG.registration(input, nsample=nsample, continuous=continuous)
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals_train(self):
        out_dict = OrderedDict()
        if self.centered:
            min_max = (-1, 1)
        else:
            min_max = (0, 1)
        out_dict['M'] = Metrics.tensor2im(self.data['M'].detach().float().cpu(), min_max=min_max)
        out_dict['F'] = Metrics.tensor2im(self.data['F'].detach().float().cpu(), min_max=min_max)
        out_dict['out_M'] = Metrics.tensor2im(self.out_M.detach().float().cpu(), min_max=(0, 1))
        out_dict['flow'] = Metrics.tensor2im(self.flow.detach().float().cpu(), min_max=min_max)
        return out_dict

    def get_current_visuals(self, sample=False):
        out_dict = OrderedDict()
        if self.centered:
            min_max = (-1, 1)
        else:
            min_max = (0, 1)
        out_dict['MF'] = Metrics.tensor2im(self.MF.detach().float().cpu(), min_max=min_max)
        out_dict['M'] = Metrics.tensor2im(self.data['M'].detach().float().cpu(), min_max=min_max)
        out_dict['F'] = Metrics.tensor2im(self.data['F'].detach().float().cpu(), min_max=min_max)
        out_dict['out_M'] = Metrics.tensor2im(self.out_M.detach().float().cpu(), min_max=(0, 1))
        out_dict['flow'] = Metrics.tensor2im(self.flow.detach().float().cpu(), min_max=min_max)
        return out_dict

    def get_current_generation(self):
        out_dict = OrderedDict()

        out_dict['MF'] = self.MF.detach().float().cpu()
        return out_dict

    def get_current_registration(self):
        out_dict = OrderedDict()

        out_dict['out_M'] =self.out_M.detach().float().cpu()
        out_dict['flow'] = self.flow.detach().float().cpu()
        out_dict['contD'] = self.contD.detach().float().cpu()
        out_dict['contF'] = self.contF.detach().float().cpu()
        return out_dict

    def print_network(self, net):
        s, n = self.get_network_description(net)
        if isinstance(net, nn.DataParallel):
            net_struc_str = '{} - {}'.format(net.__class__.__name__,
                                             net.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(net.__class__.__name__)

        logger.info(
            'Network structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_network(self, epoch, iter_step):
        genG_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_gen_G.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        # gen
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, genG_path)

        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step,
                     'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, opt_path)

        logger.info(
            'Saved model in [{:s}] ...'.format(genG_path))

    def load_network(self):
        load_path = self.opt['path']['resume_state']

        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            genG_path = '{}_gen_G.pth'.format(load_path)

            opt_path = '{}_opt.pth'.format(load_path)
            # gen
            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module
            network.load_state_dict(torch.load(
                genG_path), strict=(not self.opt['model']['finetune_norm']))

            if self.opt['phase'] == 'train':
                # optimizer
                opt = torch.load(opt_path)
                self.optG.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']
```

**__init__**函数

1. `super(DDPM, self).__init__(opt)`: 这一行调用了 `DDPM` 类的父类 `BaseModel` 的初始化方法，将 `opt` 参数传递给父类的初始化方法。
2. `self.netG = self.set_device(networks.define_G(opt))`: 这一行定义了一个名为 `netG` 的属性，它是通过调用 `networks.define_G(opt)` 来创建的网络模型，并且通过 `self.set_device()` 方法将其移动到设备上。
3. `self.schedule_phase = None`: 这一行初始化了 `schedule_phase` 属性为 `None`，用于记录当前噪声表的阶段。
4. `self.centered = opt['datasets']['centered']`: 这一行设置了 `centered` 属性，它从 `opt` 参数中获取了数据集的中心化信息。
5. `self.set_loss()`: 这一行调用了 `set_loss()` 方法，用于设置损失函数。
6. `self.set_new_noise_schedule(opt['model']['beta_schedule']['train'], schedule_phase='train')`: 这一行调用了 `set_new_noise_schedule()` 方法，用于设置噪声表的调度，并将调度阶段设置为 `'train'`。
7. `self.load_network()`: 这一行调用了 `load_network()` 方法，用于加载预训练的网络模型。
8. `if self.opt['phase'] == 'train':`: 这一行检查了训练阶段的情况，如果当前阶段是训练阶段，则执行以下代码块。
9. `self.netG.train()`: 这一行将网络模型 `netG` 设置为训练模式。
10. `if opt['model']['finetune_norm']:`: 这一行检查了是否需要对模型进行微调。
11. `optim_params = []`: 初始化一个空列表，用于存储需要优化的参数。
12. `for k, v in self.netG.named_parameters():`: 这一行遍历了网络模型 `netG` 中的所有参数及其名称。
13. `v.requires_grad = False`: 将参数的 `requires_grad` 属性设置为 `False`，表示这些参数不需要梯度更新。
14. `if k.find('transformer') >= 0:`: 这一行检查参数名称中是否包含 `'transformer'`，如果包含，则将其设置为需要优化的参数，并且将其梯度初始化为零。
15. `optim_params.append(v)`: 将需要优化的参数添加到 `optim_params` 列表中。
16. `self.optG = torch.optim.Adam(optim_params, lr=opt['train']["optimizer"]["lr"], betas=(0.5, 0.999))`: 这一行定义了优化器 `optG`，使用 Adam 优化算法，并传递了需要优化的参数列表以及学习率和动量参数。
17. `self.log_dict = OrderedDict()`: 初始化了一个有序字典 `log_dict`，用于记录训练过程中的损失值。
18. `self.print_network(self.netG)`: 这一行调用了 `print_network()` 方法，用于打印网络模型的结构信息。

**feed_data函数**

1. 迁移数据到本设备上

**optimize_parameters**函数

1. `self.optG.zero_grad()`: 将优化器 `self.optG` 中所有参数的梯度清零，以准备接收新一轮的梯度计算。
2. `score, loss = self.netG(self.data)`: 将输入数据 `self.data` 通过网络模型 `self.netG` 进行前向传播，得到模型的输出分数 `score` 和损失值 `loss`。
3. `self.score, self.out_M, self.flow = score`: 将 `score` 中的分数值分别赋给 `self.score`、`self.out_M` 和 `self.flow`。
4. `l_pix, l_sim, l_smt, l_tot = loss`: 将 `loss` 中的各个损失项分别赋给 `l_pix`、`l_sim`、`l_smt` 和 `l_tot`。
5. `l_tot.backward()`: 对总损失 `l_tot` 进行反向传播，计算参数的梯度。
6. `self.optG.step()`: 根据参数的梯度，使用优化器 `self.optG` 来更新模型的参数，执行一步优化。
7. `self.log_dict['l_pix'] = l_pix.item()`: 将各个损失项的值保存到 `self.log_dict` 中，以便记录训练过程中的损失值。`.item()` 方法用于将损失项的值转换为 Python 标量并保存。

可以看到，分数和损失都是通过`self.netG`确定的，所以需要重点关注`model/networks.py`中的network函数，其中记录了扩散的整个网络。我们先解析这个函数。

#### **define_G函数**

```python
# Generator
def define_G(opt):
    model_opt = opt['model']
    if model_opt['netDim'] == 2:
        from .diffusion_net_2D import diffusion, unet
        from .deformation_net_2D import registUnetBlock
    elif model_opt['netDim'] == 3:
        from .diffusion_net_3D import diffusion, unet
        from .deformation_net_3D import registUnetBlock
    else:
        raise('model dimension error')

    model_score = unet.UNet(
        in_channel=model_opt['unet']['in_channel'],
        out_channel=model_opt['unet']['out_channel'],
        inner_channel=model_opt['unet']['inner_channel'],
        channel_mults=model_opt['unet']['channel_multiplier'],
        attn_res=model_opt['unet']['attn_res'],
        res_blocks=model_opt['unet']['res_blocks'],
        dropout=model_opt['unet']['dropout'],
        image_size=model_opt['diffusion']['image_size']
    )

    model_field = registUnetBlock(model_opt['field']['in_channel'],
                           model_opt['field']['encoder_nc'],
                           model_opt['field']['decoder_nc'])

    netG = diffusion.GaussianDiffusion(
        model_score, model_field,
        channels=model_opt['diffusion']['channels'],
        loss_type='l2',    # L1 or L2
        conditional=model_opt['diffusion']['conditional'],
        schedule_opt=model_opt['beta_schedule']['train'],
        loss_lambda=model_opt['loss_lambda']
    )
    if opt['phase'] == 'train':
        load_path = opt['path']['resume_state']
        if load_path is None:
            init_weights(netG.denoise_fn, init_type='orthogonal')
            init_weights(netG.field_fn, init_type='normal')
    if opt['gpu_ids'] and opt['distributed']:
        assert torch.cuda.is_available()
        netG = nn.DataParallel(netG)
    return netG
```


这段代码定义了一个函数 `define_G(opt)`，该函数的作用是根据给定的选项 `opt` 来定义并返回一个生成器网络模型 `netG`。

具体步骤如下：

1. 根据选项 `opt['model']['netDim']` 的值，确定模型是二维还是三维的：
   - 如果 `netDim` 为 2，导入二维的扩散网络和 U-Net 架构以及相关的配准 U-Net 模块。
   - 如果 `netDim` 为 3，导入三维的扩散网络和 U-Net 架构以及相关的配准 U-Net 模块。
   - 否则，抛出错误提示模型维度错误。
2. 根据给定的模型选项 `model_opt` 和上一步确定的模型结构，创建 U-Net 网络模型 `model_score` 和配准 U-Net 模块 `model_field`。这些模型的参数和结构由选项中的参数配置确定。
3. 使用扩散网络 `GaussianDiffusion` 将 U-Net 网络 `model_score` 和配准 U-Net 模块(`registUnetBlock`) `model_field` 结合起来构建生成器网络模型 `netG`。这里还指定了一些参数，如通道数、损失类型、条件形式等。
4. 如果当前阶段是训练阶段（`opt['phase'] == 'train'`），并且没有提供预训练模型的路径（`opt['path']['resume_state']` 为 `None`），则对生成器网络模型的权重进行初始化。这里调用了 `init_weights` 函数来初始化网络的权重参数。
5. 如果使用了 GPU 并且进行了分布式训练（`opt['gpu_ids']` 和 `opt['distributed']` 均为真），则使用 `nn.DataParallel` 将生成器网络模型包装起来以支持多 GPU 训练。
6. 最后返回生成器网络模型 `netG`。

针对这个函数，我们主要关注其中的

```python
from .diffusion_net_2D import diffusion, unet
from .deformation_net_2D import registUnetBlock
```

两个模块中的内容，因为我们的特征提取也主要是出现在二维的场景当中的。

### **diffusion_2D/unet.py**

```python
class UNet(nn.Module):
    def __init__(
        self,
        in_channel=6,
        out_channel=3,
        inner_channel=32,
        channel_mults=(1, 2, 4, 8, 8),
        attn_res=(8),
        res_blocks=3,
        dropout=0,
        with_time_emb=True,
        image_size=128
    ):
        super().__init__()

        if with_time_emb:
            time_dim = inner_channel
            self.time_mlp = nn.Sequential(
                TimeEmbedding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            time_dim = None
            self.time_mlp = None

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, time_emb_dim=time_dim, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, time_emb_dim=time_dim,
                               dropout=dropout, with_attn=True),
            ResnetBlocWithAttn(pre_channel, pre_channel,
                               time_emb_dim=time_dim, dropout=dropout, with_attn=False)
        ])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks+1):
                ups.append(ResnetBlocWithAttn(
                    pre_channel+feat_channels.pop(), channel_mult, time_emb_dim=time_dim, dropout=dropout, with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res*2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(out_channel, in_channel))

    def forward(self, x, time):
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        feats = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)

        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(torch.cat((x, feats.pop()), dim=1), t)
            else:
                x = layer(x)
        recon = self.final_conv(x)
        return recon
```

1. 初始化函数 `__init__`：
   - 接受一系列参数，如输入通道数、输出通道数、内部通道数、通道倍数、注意力机制的使用、残差块数量、dropout 概率等。
   - 根据参数设置是否添加时间嵌入层，并定义时间嵌入层的结构。
   - 创建 U-Net 架构的各个层，包括下采样（编码器）、中间模块和上采样（解码器）。
   - 设置最终的卷积层，用于生成输出。
2. 前向传播函数 `forward`：
   - 接受输入张量 `x` 和时间信息 `time`。
   - 如果设置了时间嵌入层，则对时间信息进行嵌入操作。
   - 通过 U-Net 的下采样部分，依次对输入进行特征提取，并保存每一层的特征。
   - 对中间模块进行特征提取。
   - 通过 U-Net 的上采样部分，依次将特征进行上采样，并与相应层的下采样特征进行连接。
   - 最后通过最终的卷积层生成输出。

`__init__`函数

```python
if with_time_emb:
            time_dim = inner_channel
            self.time_mlp = nn.Sequential(
                TimeEmbedding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            time_dim = None
            self.time_mlp = None
```

定义一个时间嵌入层 `time_mlp`，作用是将时间信息嵌入到模型当中，处理有时间特征的数据。

- 如果 `with_time_emb` 为真（即为 `True`），则表示需要在模型中添加时间嵌入层。在这种情况下，首先确定了时间嵌入层的维度 `time_dim`，其维度与模型的内部通道数 `inner_channel` 相同。
- 然后使用 `nn.Sequential` 构建一个包含多个层的序列，其中包括一个时间嵌入层 `TimeEmbedding`、两个线性层 `nn.Linear` 和一个激活函数 `Swish`。这些层的作用是将时间信息转换为与模型特征相匹配的形式。
- 如果 `with_time_emb` 为假（即为 `False`），则表示不需要添加时间嵌入层。在这种情况下，将 `time_dim` 设置为 `None`，并且将 `self.time_mlp` 设置为 `None`。

**TimeEmbedding**

这里的时间嵌入层`TimeEmbedding`在这个函数里这样定义：

```python
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) *
            (-math.log(10000) / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, input):
        shape = input.shape
        sinusoid_in = torch.ger(input.view(-1).float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)
        return pos_emb
```

`init`函数

- `def __init__(self, dim):`：初始化函数，接收一个参数 `dim`，表示嵌入维度。
- `super().__init__()`：调用父类（`nn.Module`）的初始化函数。
- `self.dim = dim`：将输入的维度保存在类属性 `dim` 中。
- `inv_freq = torch.exp(...)`：计算正弦和余弦函数的频率，这里使用了固定的频率计算方法。
- `torch.arange(0, dim, 2, dtype=torch.float32)`：创建一个从 0 到 `dim-1` 的等差数列，步长为 2，数据类型为 `torch.float32`，这些数字将用作频率的指数。
- `(-math.log(10000) / dim)`：计算用于缩放频率的常数，这个常数与嵌入维度相关。
- `self.register_buffer("inv_freq", inv_freq)`：将频率作为缓冲区（buffer）注册到模型中，这样它们就成为了模型的一部分。

`forward`函数

- `shape = input.shape`：获取输入张量的形状。
- `torch.ger(input.view(-1).float(), self.inv_freq)`：将输入张量展平为一维，并将频率与其进行张量积，这一步实现了将频率应用于每个时间步长的功能。
- `sinusoid_in.sin()` 和 `sinusoid_in.cos()`：分别计算正弦和余弦函数的值。
- `torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)`：在最后一个维度上拼接正弦和余弦函数的结果。
- `pos_emb = pos_emb.view(*shape, self.dim)`：将拼接后的结果变换回原始形状。
- `return pos_emb`：返回时间嵌入的张量。

具体的时间计算频率的公式如下所示：
$$
freq(i)=exp(\frac{−log(10000)}{d}⋅i)
$$
这样就把时间信息嵌入到输入当中去。

继续回到`Unet`的网络架构中去。

```python
num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
```

1. `num_mults = len(channel_mults)`：计算 `channel_mults` 列表的长度，即通道倍数的总长度。
2. `pre_channel = inner_channel`：将 `pre_channel` 初始化为 `inner_channel`，这是下一个通道数的初始值。
3. `feat_channels = [pre_channel]`：创建一个包含初始通道数 `pre_channel` 的列表 `feat_channels`。
4. `now_res = image_size`：将当前分辨率初始化为 `image_size`。
5. `downs = [nn.Conv2d(in_channel, inner_channel, kernel_size=3, padding=1)]`：创建一个包含第一个卷积层的列表 `downs`，该卷积层将输入通道数为 `in_channel` 的特征图转换为通道数为 `inner_channel` 的特征图。

```python
for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, time_emb_dim=time_dim, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2
        self.downs = nn.ModuleList(downs)
```

1. `for ind in range(num_mults):`：遍历 `num_mults` 次，其中 `num_mults` 是 `channel_mults` 列表的长度。
2. `is_last = (ind == num_mults - 1)`：检查是否是最后一个循环迭代。
3. `use_attn = (now_res in attn_res)`：检查当前分辨率是否在关注分辨率列表 `attn_res` 中。
4. `channel_mult = inner_channel * channel_mults[ind]`：根据当前索引 `ind` 从 `channel_mults` 中获取通道数的倍增系数。
5. `for _ in range(0, res_blocks):`：循环 `res_blocks` 次，其中 `res_blocks` 是残差块的数量。
6. `downs.append(ResnetBlocWithAttn(pre_channel, channel_mult, time_emb_dim=time_dim, dropout=dropout, with_attn=use_attn))`：向 `downs` 列表中添加一个残差块，其中包含当前通道数 `pre_channel` 和计算得到的倍增后的通道数 `channel_mult`。
7. `feat_channels.append(channel_mult)`：向 `feat_channels` 列表中添加当前倍增后的通道数，以便后续的上采样操作使用。
8. `pre_channel = channel_mult`：更新 `pre_channel` 为当前通道数的倍增后的值，以便下一个残差块使用。
9. `if not is_last:`：如果不是最后一个循环迭代：
10. `downs.append(Downsample(pre_channel))`：向 `downs` 列表中添加一个下采样模块，用于降低特征图的分辨率。
11. `feat_channels.append(pre_channel)`：向 `feat_channels` 列表中添加当前通道数，以便后续的上采样操作使用。
12. `now_res = now_res//2`：更新当前分辨率为原来的一半，以反映下采样的影响。
13. `self.downs = nn.ModuleList(downs)`：将 `downs` 列表转换为 `nn.ModuleList` 类型，并将其保存在模型中。

整体来说，这段代码执行了一些任务，使用`num_mults`控制下采样的层数。

**通过将输入特征图的通道数×倍增系数来增加特征图的通道数，这个倍增系数就是channel_mults。每个下采样层内调用三次注意力的残差块，之后再加入一个下采样快。整合到一起变成一个下采样层。**

注意力的残差块是这个类`ResnetBlocWithAttn`

#### **ResnetBlocWithAttn**

```python
class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, time_emb_dim, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if(self.with_attn):
            x = self.attn(x)
        return x
```

1. `ResnetBlocWithAttn` 接受输入特征 `x` 和时间嵌入 `time_emb`，其中 `dim` 是输入特征的维度，`dim_out` 是输出特征的维度。
2. 在初始化时，根据参数 `with_attn` 的取值，决定是否添加注意力机制。如果 `with_attn` 为 True，则在残差块后添加自注意力机制。
3. `ResnetBlocWithAttn` 内部包含一个 `ResnetBlock`，用于实现残差连接和特征变换。
4. 在前向传播过程中，首先将输入特征 `x` 通过残差块 `res_block` 进行处理，得到残差连接后的输出特征。
5. 如果 `with_attn` 为 True，则将输出特征传递给自注意力模块 `attn` 进行加权处理，以增强特征表示的表达能力。
6. 返回经过残差连接和注意力机制处理后的特征表示。

残差快的函数比较好理解，这里主要关注一下自注意力机制块：

#### **SelfAttention**

```python
class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1):
        super().__init__()
        self.n_head = n_head
        self.norm = nn.GroupNorm(32, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input
```

这里定义了三个层，组归一化层，卷积层和输出层

1. 组归一化（Group Normalization）层 `norm`，用于对输入特征进行归一化。
2. 卷积层 `qkv`，用于生成查询（query）、键（key）和值（value）。它使用 `nn.Conv2d` 表示在二维特征图上进行卷积操作，输入通道数为 `in_channel`，输出通道数为 `in_channel * 3`，卷积核大小为1×1。
3. 卷积层 `qkv`，用于生成查询（query）、键（key）和值（value）。它使用 `nn.Conv2d` 表示在二维特征图上进行卷积操作，输入通道数为 `in_channel`，输出通道数为 `in_channel * 3`，卷积核大小为1×1。

接收注意力头的数量`self.n_head`

每个头的维度 `head_dim`，其中 `head_dim` 等于输入通道数除以头的数量。

这里通过卷积层 `qkv` 对归一化后的特征进行卷积操作，并将输出的特征重塑成形状为 `(batch, n_head, head_dim * 3, height, width)` 的张量。

使用 `chunk()` 方法将卷积后的特征张量沿着通道维度切分成三份，分别代表查询、键和值。

这里使用 Einstein Summation (`torch.einsum`) 对查询和键进行点积操作，得到注意力分数，并进行归一化。`bnchw` 和 `bncyx` 表示张量的维度顺序，`bhwyx` 表示输出的注意力分数的维度顺序。

`attn = attn.view(batch, n_head, height, width, -1)`将注意力分数张量重塑成形状为 `(batch, n_head, height, width, height * width)` 的张量。

这里对注意力分数进行 Softmax 操作，得到注意力权重。

这一行将注意力权重张量重塑成形状为 `(batch, n_head, height, width, height, width)` 的张量。

这里使用 Einstein Summation (`torch.einsum`) 将注意力权重与值相乘，得到加权后的特征表示。

这里通过输出卷积层 `out`，将加权后的特征表示映射回原始特征维度。

最后，将输出的加权特征与输入特征进行残差连接，得到最终的输出。

**这个`U-net.py`里面好像就是一些比较统一的网络结构。**

### model/deformation_net_2D.py

### registUnetBlock模块

这是一个变形场的模块，暂时先不关注这个模块。

```python
class registUnetBlock(nn.Module):
    def __init__(self, input_nc, encoder_nc, decoder_nc):
        super(registUnetBlock, self).__init__()
        self.inconv = inblock(input_nc, encoder_nc[0], stride=1)
        self.downconv1 = downblock(encoder_nc[0], encoder_nc[1])
        self.downconv2 = downblock(encoder_nc[1], encoder_nc[2])
        self.downconv3 = downblock(encoder_nc[2], encoder_nc[3])
        self.downconv4 = downblock(encoder_nc[3], encoder_nc[4])
        self.upconv1 = upblock(encoder_nc[4], encoder_nc[4]+encoder_nc[3], decoder_nc[0])
        self.upconv2 = upblock(decoder_nc[0], decoder_nc[0]+encoder_nc[2], decoder_nc[1])
        self.upconv3 = upblock(decoder_nc[1], decoder_nc[1]+encoder_nc[1], decoder_nc[2])
        self.keepblock = CRblock(decoder_nc[2], decoder_nc[3])
        self.upconv4 = upblock(decoder_nc[3], decoder_nc[3]+encoder_nc[0], decoder_nc[4])
        self.outconv = outblock(decoder_nc[4], decoder_nc[5], stride=1)
        self.spatialtransform = Dense2DSpatialTransformer()

    def forward(self, input):
        x1 = self.inconv(input)
        x2 = self.downconv1(x1)
        x3 = self.downconv2(x2)
        x4 = self.downconv3(x3)
        x5 = self.downconv4(x4)
        x = self.upconv1(x5, x4)
        x = self.upconv2(x, x3)
        x = self.upconv3(x, x2)
        x = self.keepblock(x)
        x = self.upconv4(x, x1)
        flow = self.outconv(x)
        mov = (input[:, :1] + 1) / 2.0
        out = self.spatialtransform(mov, flow)
        return out, flow
```

### model/deformation_net_2D/diffusion.py

这个类是diffusion的主要类

主要定义在`GaussianDiffusion`类当中。

```python
class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn, field_fn,  # 去噪函数和场函数
        channels=3,  # 通道数，默认为3
        loss_type='l1',  # 损失类型，默认为l1
        conditional=True,  # 是否条件化，默认为True
        schedule_opt=None,  # 时间表选项，默认为None
        loss_lambda=1  # 损失权重，默认为1
    ):
        super().__init__()  # 调用父类的初始化方法
        self.channels = channels  # 通道数
        self.denoise_fn = denoise_fn  # 去噪函数
        self.field_fn = field_fn  # 场函数
        self.conditional = conditional  # 是否条件化
        self.loss_type = loss_type  # 损失类型
        self.lambda_L = loss_lambda  # 损失权重
        if schedule_opt is not None:  # 如果时间表选项不为空，则执行以下内容
            pass

    def set_loss(self, device):  # 设置损失函数
        if self.loss_type == 'l1':  # 如果损失类型为l1
            self.loss_func = nn.L1Loss(reduction='mean').to(device)  # 使用L1损失函数
        elif self.loss_type == 'l2':  # 如果损失类型为l2
            self.loss_func = nn.MSELoss(reduction='mean').to(device)  # 使用MSE损失函数
        else:  # 如果损失类型未实现
            raise NotImplementedError()  # 抛出未实现错误
        self.loss_ncc = loss.crossCorrelation2D(1, kernel=(9, 9)).to(device)  # 设置归一化互相关损失
        self.loss_reg = loss.gradientLoss("l2").to(device)  # 设置梯度损失函数

    def set_new_noise_schedule(self, schedule_opt, device):  # 设置新的噪声时间表
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)  # 转换为torch张量的函数
        betas = make_beta_schedule(  # 创建噪声时间表的函数
            schedule=schedule_opt['schedule'],  # 时间表类型
            n_timestep=schedule_opt['n_timestep'],  # 时间步数
            linear_start=schedule_opt['linear_start'],  # 线性起始值
            linear_end=schedule_opt['linear_end'])  # 线性结束值
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas  # 将张量转换为numpy数组
        alphas = 1. - betas  # 计算alphas
        alphas_cumprod = np.cumprod(alphas, axis=0)  # 计算alphas的累积乘积
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])  # 计算alphas_cumprod的前一个值

        timesteps, = betas.shape  # 获取时间步数
        self.num_timesteps = int(timesteps)  # 将时间步数转换为整数
        self.register_buffer('betas', to_torch(betas))  # 注册betas张量
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))  # 注册alphas_cumprod张量
        self.register_buffer('alphas_cumprod_prev',  # 注册alphas_cumprod_prev张量
                             to_torch(alphas_cumprod_prev))

        # 计算扩散过程的参数
        self.register_buffer('sqrt_alphas_cumprod',  # 注册sqrt_alphas_cumprod张量
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',  # 注册sqrt_one_minus_alphas_cumprod张量
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',  # 注册log_one_minus_alphas_cumprod张量
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',  # 注册sqrt_recip_alphas_cumprod张量
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',  # 注册sqrt_recipm1_alphas_cumprod张量
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # 计算后验概率
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)  # 计算后验方差
        # 以上：等于1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',  # 注册posterior_variance张量
                             to_torch(posterior_variance))
        # 以下：修剪log计算，因为后验方差在扩散链的开始时为0
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(  # 注册posterior_mean_coef1张量
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(  # 注册posterior_mean_coef2张量
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def q_mean_variance(self, x_start, t):  # 计算q分布的均值和方差
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start  # 计算均值
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)  # 计算方差
        log_variance = extract(  # 计算对数方差
            self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):  # 从噪声预测初始值
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):  # 计算后验分布
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(  # 修剪后验对数方差
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):  # 计算先验分布的均值和方差
        score = self.denoise_fn(torch.cat([condition_x, x], dim=1), t)  # 计算评分
        x_recon = self.predict_start_from_noise(x, t=t, noise=score)  # 从噪声预测初始值

        if clip_denoised:  # 如果裁剪去噪值
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(  # 计算后验均值、方差和对数方差
            x_start=x_recon, x_t=x, t=t)

        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False, condition_x=None):  # 从先验分布中采样
        b, *_, device = *x.shape, x.device  # 获取张量形状和设备
        model_mean, _, model_log_variance = self.p_mean_variance(  # 计算先验均值、方差和对数方差
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        noise = noise_like(x.shape, device, repeat_noise)  # 生成与x形状相同的噪声张量
        # t == 0时无噪声
        nonzero_mask = (1 - (t == 0).float()).reshape(b,
                                                      *((1,) * (len(x.shape) - 1)))  # 非零噪声掩码
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise  # 返回采样值

    # @torch.no_grad()
    def p_sample_loop(self, x_in, continuous=False):  # 从先验分布中循环采样
        device = self.betas.device  # 获取设备

        x = x_in  # 输入张量
        x_m = x_in[:, :1]  # 输入张量的前一半
        shape = x_m.shape  # 输入张量的形状
        b = shape[0]  # 批次大小

        fw_timesteps = 200  # 正向时间步数
        bw_timesteps = 200  # 反向时间步数
        t = torch.full((b,), fw_timesteps, device=device, dtype=torch.long)  # 充满正向时间步数的张量
        with torch.no_grad():  # 不进行梯度计算
            # 正向采样
            d2n_img = self.q_sample(x_m, t)

            # 反向采样
            img = d2n_img  # 图像张量
            ret_img = d2n_img  # 返回图像张量

            for ispr in range(1):  # 循环采样
                for i in (reversed(range(0, bw_timesteps))):  # 反向循环
                    t = torch.full((b,), i, device=device, dtype=torch.long)  # 充满反向时间步数的张量
                    img = self.p_sample(img, t, condition_x=x)  # 从先验分布中采样

                    if i % 11 == 0:  # 每隔11个时间步骤
                        ret_img = torch.cat([ret_img, img], dim=0)  # 连接图像张量

        if continuous:  # 如果连续
            return ret_img  # 返回图像张量
        else:
            return ret_img[-1:]  # 返回最后一个图像张量

    # @torch.no_grad()
    def generation(self, x_in, continuous=False):  # 生成图像
        return self.p_sample_loop(x_in, continuous)  # 调用p_sample_loop方法进行采样

    # @torch.no_grad()
    def registration(self, x_in, nsample=7, continuous=False):  # 注册
        x_m = x_in[:, :1]  # 输入图像的前一半
        x_f = x_in[:, 1:]  # 输入图像的后一半
        eta = np.linspace(0, 1, nsample)  # 生成nsample个等距数
        b, c, h, w = x_m.shape  # 输入图像的形状
        cont_deform = x_m  # 连续变形张量
        cont_field = torch.zeros((b, 2, h, w), device=self.betas.device)  # 连续场张量
        with torch.no_grad():  # 不进行梯度计算
            t = torch.full((x_in.shape[0],), 0, device=self.betas.device, dtype=torch.long)  # 充满零的张量
            score = self.denoise_fn(torch.cat([x_in, x_f], dim=1), t)  # 计算去噪分数
        for ieta in range(nsample):  # 循环遍历nsample
            score_eta = score * eta[ieta]  # 分数乘以eta值
            deform, flow = self.field_fn(torch.cat([x_m, score_eta], dim=1))  # 计算变形和场
            cont_deform = torch.cat([cont_deform, deform], dim=0)  # 连接连续变形张量
            cont_field = torch.cat([cont_field, flow], dim=0)  # 连接连续场张量

        if continuous:  # 如果连续
            return deform, flow, cont_deform[1:], cont_field[1:]  # 返回连续变形、场、变形张量和场张量
        else:
            return deform, flow, cont_deform[-1], cont_field[-1]  # 返回最后一个变形张量和场张量

    def q_sample(self, x_start, t, noise=None):  # 从q分布中采样
        noise = default(noise, lambda: torch.randn_like(x_start))  # 默认噪声为标准正态分布噪声

        # # fix gama
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod,
                    t, x_start.shape) * noise
        )

    def p_losses(self, x_in, noise=None):  # 计算损失
        x_start = x_in['F']  # 初始图像
        [b, c, h, w] = x_start.shape  # 图像形状
        t = torch.randint(0, self.num_timesteps, (b,),  # 随机生成时间步数
                          device=x_start.device).long()

        noise = default(noise, lambda: torch.randn_like(x_start))  # 默认噪声为标准正态分布噪声
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)  # 从q分布中采样噪声

        x_recon = self.denoise_fn(torch.cat([x_in['M'], x_in['F'], x_noisy], dim=1), t)  # 计算去噪图像
        l_pix = self.loss_func(noise, x_recon)  # 计算像素损失

        output, flow = self.field_fn(torch.cat([x_in['M'], x_recon], dim=1))  # 计算场输出
        l_sim = self.loss_ncc(output, x_in['F']) * self.lambda_L  # 计算归一化互相关损失
        l_smt = self.loss_reg(flow) * self.lambda_L  # 计算梯度损失
        loss = l_pix + l_sim + l_smt  # 总损失

        return [x_recon, output, flow], [l_pix, l_sim, l_smt, loss]  # 返回重建图像、输出和场以及各项损失

    def forward(self, x, *args, **kwargs):  # 正向传播
        return self.p_losses(x, *args, **kwargs)  # 返回损失

```

