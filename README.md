# CGH Simulation Platform：角谱理论 + Gerchberg–Saxton 纯相位计算全息仿真平台

## 1. 项目简介

一个基于严格角谱法（Angular Spectrum Method, ASM，无近轴近似）与经典 Gerchberg–Saxton (GS) 迭代算法，从零手写实现的纯相位计算全息图（CGH）设计与仿真平台。技术栈：Python 3.9+ / NumPy / SciPy / Matplotlib，不依赖任何第三方光学计算库，所有衍射传播与相位恢复公式均为底层实现。平台内置倏逝波截断与带限角谱（BLAS）抗混叠滤波，可配置波长、采样、传播距离等物理参数，并输出论文级四联图与收敛曲线，用于验证多焦点阵列与涡旋光束等复杂光场的纯相位全息重建能力。

## 2. 物理原理

### 2.1 角谱衍射理论

标量光场 $U(x,y;0)$ 经自由空间传播距离 $z$ 后：

$$U(x,y;z) = \mathcal{F}^{-1}\Big\{\, \mathcal{F}\{U(x,y;0)\} \cdot H(f_x,f_y;z) \,\Big\}$$

角谱传递函数（不做菲涅尔近轴近似，严格保留根号项）：

$$H(f_x,f_y;z) = \exp\!\Big[\, j\,2\pi z \sqrt{\tfrac{1}{\lambda^2} - f_x^2 - f_y^2}\, \Big], \quad f_x^2+f_y^2 < \tfrac{1}{\lambda^2}$$

**倏逝波截断**：当 $f_x^2+f_y^2 \ge 1/\lambda^2$ 时根号项变为虚数，若不处理会使 $H$ 变成随 $z$ 指数增长的因子，导致数值发散——因此该频段直接强制 $H=0$。

**带限角谱滤波（BLAS）**：即便滤除倏逝波，$H$ 仍是一个啁啾（chirp）相位函数，其局部空间频率随 $z$ 增大而增大；当仿真窗口 $L_x=N\,dx$ 有限时，该啁啾很容易在到达倏逝波边界之前就先超过网格的奈奎斯特频率，产生频谱混叠伪影（远距离传播或欠采样时尤其明显）。参照 Matsushima & Shimobaba (2009) 的方法，对传递函数额外施加频域矩形窗：

$$f_{x,\text{limit}} = \frac{1}{\lambda\sqrt{(2z/L_x)^2+1}}, \qquad |f_x| < f_{x,\text{limit}}\ (\text{同理 } f_y)$$

`figures/blas_aliasing_demo.png` 直接对比了同一远距离传播（z = 120 mm）在开启/关闭 BLAS 时的重建强度：关闭时全画幅出现棋盘状混叠伪影，开启后伪影被显著抑制。

### 2.2 Gerchberg–Saxton 相位恢复流程

```
初始化全息面相位 φ_holo (全零 或 随机)
for iteration in 1..N_iter:
    1. U_holo = 1 · exp(jφ_holo)                 # 纯相位 SLM 约束：振幅恒为 1
    2. U_target = ASM_propagate(U_holo, +z)       # 正向传播到目标面
    3. φ_target = angle(U_target)                 # 保留目标面相位（自由变量）
    4. U_target' = A_target · exp(jφ_target)      # 用目标振幅替换重建振幅
    5. U_holo_new = ASM_propagate(U_target', -z)  # 反向传播回全息面
    6. φ_holo = angle(U_holo_new)                 # 丢弃振幅，只保留相位 -> 下一轮迭代
```

其中步骤 1、6 是纯相位 SLM 硬件约束的直接体现——真实相位型 LCOS/SLM 逐像素只能延迟光程、不能调制振幅，因此每轮迭代都必须把全息面振幅重新归一化为常数。

## 3. 工业应用场景

- **AR 衍射光波导**：多焦点点阵测试（`generate_dot_array`）直接对应波导出瞳光栅、多点光场耦合的设计验证场景。
- **车载 HUD**：远距离虚像投影对传播距离 $z$ 远大于孔径尺寸的场景高度敏感，正是 BLAS 抗混叠滤波要解决的典型工况。
- **SLM 动态光束整形**：涡旋光束（`generate_vortex_beam`）用于验证平台在复杂、非凸目标光场（含暗核环形结构）下的相位恢复鲁棒性，对应结构光照明、光镊等应用中的动态光场调控需求。

## 4. 运行方法

```bash
# 依赖
pip install numpy scipy matplotlib

# 运行（在项目根目录）
python main.py
```

运行后会在 `./figures/` 目录下生成全部结果图，控制台会打印每组实验的最终 MSE。

## 5. 仿真参数示例（`main.py` 顶部 `CONFIG`）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `wavelength` λ | 532 nm | 绿光激光 |
| `N` | 256 × 256 | 采样网格 |
| `dx` | 8 μm | 采样步长（典型 LCOS 像素间距） |
| `z` | 10 mm | 全息面→目标面传播距离 |
| `num_iterations` | 60 | GS 迭代次数 |
| `use_band_limit` | True | 是否启用 BLAS 滤波 |
| `dot_grid_size` | 5×5 | 点阵光斑数量 |
| `vortex_topological_charge` | l = 3 | 涡旋光拓扑荷（环形靶标几何参数） |

全部参数集中在 `main.py` 的 `CONFIG` 字典中，一处修改即可重新运行全部实验。

## 6. 结果说明

`python main.py` 会生成 5 张图：

- `blas_aliasing_demo.png` —— 同一远距离传播下，关闭/开启 BLAS 的强度分布对比（log 尺度），直观展示带限滤波对混叠伪影的抑制效果。
- `dot_array_zero_init.png` / `vortex_beam_zero_init.png` —— 标准四联图：(a) 目标光强 (b) 全息相位（SLM 加载图）(c) 重建光强（峰值归一化，便于与目标对比形状）(d) MSE 收敛曲线（log 坐标）。两组实验最终 MSE 均收敛至 $10^{-9}$ ～ $10^{-11}$ 量级（能量归一化定义，见 `gs_algorithm.compute_mse`），点阵与环形靶标的形状均被清晰重建；涡旋靶标由于其非凸环状结构与更高的空间频率含量，全息相位呈现更明显的散斑状精细结构，符合相位恢复问题的一般规律。
- `dot_array_init_comparison.png` / `vortex_beam_init_comparison.png` —— 零相位初始化 vs 随机相位初始化的收敛曲线对比。在本平台的两组默认实验中，零相位初始化的最终 MSE 略优于随机初始化；随机初始化在迭代前期下降更快，但两者的收敛特性差异整体较小，与经典 GS 文献报道的定性趋势一致（初始化策略主要影响是否落入某些局部极小值，而非绝对收敛能力的系统性差异）。

## 7. 拓展方向（面试谈资）

- **算法层面**：从标准 GS 升级到加权 GS (WGS) 或 Fienup 混合输入输出 (HIO) 算法，解决标准 GS 在多点阵重建中常见的局部最优 / 光斑非均匀 / 散斑噪声问题——`gs_algorithm.py` 中已预留 `apply_weighted_amplitude()` 与 `apply_hio_feedback()` 两个接口（含完整算法说明的 docstring），可直接在 GS 主循环的步骤 4 / 步骤 6 处接入。
- **物理模型层面**：引入真实 SLM 的像素化结构、填充因子（fill factor）与黑矩阵衍射效应，使仿真更贴近实际器件的衍射效率与串扰特性。
- **工程加速层面**：用 `numba` JIT 或 `cupy` GPU 加速二维 FFT 迭代循环，支撑 4K 分辨率全息图的近实时计算，服务于动态全息显示场景。

## 8. 项目结构

```
CGH_Simulation_Platform/
├── main.py                   # 主程序入口：参数配置（CONFIG）、实验调度
├── propagation.py            # 角谱法正/逆向传播（含倏逝波截断、BLAS 滤波）
├── gs_algorithm.py           # GS 迭代相位恢复核心算法（含纯相位 SLM 约束、MSE、WGS/HIO 接口）
├── targets.py                # 目标光场生成（点阵、涡旋光、BLAS 演示用圆孔）
├── visualization.py          # 论文级绘图（四联图、初始化对比图、BLAS 对比图）
├── figures/                  # 运行后生成的结果图
└── README.md
```
