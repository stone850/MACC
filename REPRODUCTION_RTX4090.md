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

## 冻结数据上的阶段三基线

`src/run_offline.py` 只支持阶段三的两条直接离线化基线。两者都从随机初始化
开始，从 `OfflineEpisodeDataset` 采样完整 episode；训练循环不会创建环境，环境只在
`offline_eval_interval` 指定的评估点以 `test_mode=True` 使用。

运行 Offline QMIX：

```bash
cd pymarl-master
python src/run_offline.py --config=qmix with \
  dataset_path=datasets/lbf/macc_medium_2m_seed1_2k_v3 \
  offline_updates=50000 use_cuda=True seed=1
```

运行保留原始高斯子任务表示、注意力和 `qmix_hidden` mixer 的 Offline MACC-QMIX：

```bash
cd pymarl-master
python src/run_offline.py --config=macc with \
  dataset_path=datasets/lbf/macc_medium_2m_seed1_2k_v3 \
  offline_updates=50000 batch_size=32 lr=0.000001 \
  target_update_interval=200 \
  offline_eval_interval=1000 offline_eval_episodes=40 \
  offline_save_interval=5000 log_interval=100 \
  use_cuda=True seed=1 \
  remarks=offline_macc_medium_2k_v3_50k
```

每次运行在 `results/offline/<unique-token>/` 下写入耐中断的
`training_metrics.jsonl`、最终 `summary.json`、TensorBoard 指标和周期 checkpoint。
公共诊断包括 `td_loss`、`q_data_mean`、`q_max_mean`、`q_tot_abs_max`、
`grad_norm`、`ood_action_rate` 和评估回报；MACC 还记录
`representation_loss`、`recon_loss` 与 `sim_loss`。`ood_action_rate` 表示有效
agent-timestep 上当前贪心动作与冻结数据动作不一致的比例，是策略分歧代理，不是行为
策略的条件支撑概率。

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
