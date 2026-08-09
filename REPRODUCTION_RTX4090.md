# 在 RTX 4090 上复现 MACC

该分支保持 MACC 算法和论文公布的超参数不变，仅更新了在 RTX 4090 上运行这份
2022 年代码所需的 Python/PyTorch 兼容性部分。此外，该分支会正确关闭
TensorBoard，并从 Sacred 入口正常返回，从而保证短时间运行也能将指标写入磁盘，
并获得 `COMPLETED` 状态。

## 固定版本

- MACC 提交：`75d7318032b0ee8a4c159646fd78e1283f248b22`
- SMAC 提交：`0de603e6a67d867b3b13582dfa5aebf36cab7f96`
- StarCraft II: `4.6.2.69232`
- Conda 环境：`macc`
- CUDA 运行时：`11.8`

## 创建环境

在仓库根目录下执行以下命令，并清除旧项目遗留的环境变量和本地代理变量：

```bash
unset LD_LIBRARY_PATH PYTHONPATH
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
conda env create -f environment-rtx4090.yml
conda activate macc
bash scripts/repro/setup_smac.sh
```

环境文件中的 pip 部分会安装该仓库内修改过的 LBF 包。SMAC 会以固定提交的归档包
形式下载，并进行校验和验证；随后归档包会被解压到 Git 忽略的 `3rdparty` 目录，
再从本地源码安装。这样，即使下载失败，也可以直接重试，而无须重新创建 Conda
环境。

## StarCraft II

将 SC2 `4.6.2.69232` 安装到：

```text
pymarl-master/3rdparty/StarCraftII
```

将 SMAC v1 地图目录放置到：

```text
pymarl-master/3rdparty/StarCraftII/Maps/SMAC_Maps
```

激活 `macc` 环境后，设置环境路径：

```bash
export SC2PATH="$PWD/pymarl-master/3rdparty/StarCraftII"
```

## 冒烟测试

每次冒烟测试训练 20,000 个环境时间步，并且每 5,000 个时间步评估一次：

```bash
bash scripts/repro/smoke.sh foraging 1
bash scripts/repro/smoke.sh pp 1
bash scripts/repro/smoke.sh sc2 1 5m_vs_6m
```

Sacred 和 TensorBoard 结果会写入 `pymarl-master/results/` 目录。Foraging 和 PP
记录 `test_return_mean`，SC2 记录 `test_battle_won_mean`。

## 兼容范围

公开代码与论文在部分 LBF 细节上存在差异。该分支以公开代码为准：视野范围
`sight` 为 `2`，每步惩罚为 `0.002`；除非冒烟测试显式覆盖，否则各环境的评估
回合数保持原 YAML 文件中的设置。

验证完成后，导出精确的环境记录：

```bash
mkdir -p artifacts
conda list --explicit -n macc > artifacts/conda-macc-explicit.txt
conda env export -n macc > artifacts/conda-macc.yml
```
