## 总体路线

基于 MACC 源码开展 OCSBC 实验时，最重要的是**不要一开始就把离线数据、保守正则、实体记忆和对比学习全部加进去**。否则最终结果异常时，很难判断问题来自哪一部分。

按照项目书，应沿着下面的单一主线推进：

$$
\boxed{
\begin{aligned}
&\text{复现 Online QMIX 与 MACC-QMIX}\\
&\rightarrow \text{建立冻结离线数据管线}\\
&\rightarrow \text{实现 Offline QMIX 与 Offline MACC}\\
&\rightarrow \text{实现 CFCQL-QMIX 与 Conservative Offline MACC}\\
&\rightarrow \text{逐步替换子任务表示}\\
&\rightarrow \text{加入跨视角对比和可见性权重}\\
&\rightarrow \text{形成完整 OCSBC-QMIX}.
\end{aligned}
}
$$

正式主干始终使用 **QMIX**。VDN 不是必经阶段，只在 QMIX 出现难以定位的稳定性问题时作为临时诊断工具。项目书也要求先保持原始 MACC 的环境、RNN、注意力和 mixer 不变，再逐步加入固定数据、保守目标和新信念模块。

---

# 一、开始编码前，先固定五项技术口径

开始修改源码前，建议先写一份一页左右的 `method_spec.md`，固定以下内容，之后不要随实验随意改动。

## 1. 正式方法是 OCSBC-QMIX

统一配置：

```yaml
mixer: qmix
```

需要实现的正式方法链是：

$$
\text{Offline QMIX}
\rightarrow
\text{CFCQL-QMIX}
\rightarrow
\text{Offline MACC-QMIX}
\rightarrow
\text{Conservative Offline MACC}
\rightarrow
\text{OCSBC-QMIX}.
$$

这个链条分别验证：

- 普通 QMIX 直接离线化会发生什么；

- 保守正则是否有效；

- MACC 的原始任务结构是否有效；

- 在相同保守骨干下，原始 MACC 表示能做到什么；

- 新的跨视角子任务信念是否提供额外收益。


## 2. 第一版必须假设实体对应关系已知

必须保证在同一个 episode 中：

$$
\text{局部观测中的第 }j\text{ 个子任务}
\longleftrightarrow
\text{全局状态中的同一物理子任务 }j.
$$

例如在 LBF 中，一个食物被采集后不能让后面的食物自动前移补槽；应保留原槽，并设置：

```python
subtask_mask[..., j] = 0
```

否则对比学习的正样本会错配。

第一版不同时研究未知子任务发现、无实体 ID 匹配或在线聚类。项目书将“可靠的子任务实例对应关系”列为方法适用前提。

## 3. 表示问题和离线价值问题分开处理

模块分工固定为：

|问题|处理模块|
|---|---|
|局部信息不足、子任务混淆|实体记忆、全局教师、重构、跨视角对比|
|应关注哪个子任务|子任务注意力|
|离线动作价值高估|CFCQL 式反事实保守正则|
|联合价值分解|QMIX|

不要用“表示不确定性”代替数据支撑度，也不要在第一版加入支撑感知注意力。项目书明确要求注意力只建模任务相关性，离线覆盖问题交给保守价值学习。

## 4. 重构目标以项目书当前公式为准

当前项目书定义：

$$
g_j^t = E_{\mathrm{global}}(u_j^{1:t})
$$

$$
\mathcal L_{\mathrm{rec}}
=
\sum_{i,j} w_{ij}^{t}
\left\|
D(z_{ij}^{t}) - \operatorname{sg}(g_j^t)
\right\|^2.
$$

因此第一版实现时：

- 局部信念 $z_{ij}^t$ 重构停止梯度的教师表示 $g_j^t$；

- $z$ 与 $g$ 同时进入独立投影头计算对比损失；

- 全局教师、解码器和投影头只在训练阶段存在。


此前讨论过直接重构原始状态 $u_j^t$ 的备选方案，但在开始编码前必须二选一。既然目前以项目书为准，就先按重构 $g_j^t$ 实现，不要两种目标混用。

## 5. 所有新增模块必须能单独关闭

至少配置以下开关：

```yaml
use_offline_training: true
use_cf_regularization: true
use_deterministic_belief: true
use_reconstruction: true
use_contrastive: true
use_visibility_weight: true
```

这样才能形成可靠消融，而不是为每个版本维护一套彼此不同的代码。

---

# 二、第一阶段：只复现原始代码

## 目标

先证明原始仓库、环境、训练流程和评估流程都是正确的。这个阶段不要引入任何离线训练或新表示模块。

建议先使用 **LBF**，而不是直接上 SMAC。原因不是把 LBF 当作最终主环境，而是 LBF 中食物实体的身份、位置、可见性和完成状态更容易检查。项目书将 LBF/Predator-Prey 定位为机制诊断环境，将 SMAC 定位为主要性能环境。

## 需要跑通的两个方法

### 1. Online QMIX

作用：

- 验证环境；

- 验证 PyMARL 的 episode 展开；

- 验证 Double Q、target network 和 QMIX mixer；

- 建立无任务表示的在线参考。


### 2. Online MACC-QMIX

作用：

- 验证原始 MACC 子任务表示；

- 验证子任务注意力；

- 验证表示损失和 TD 损失联合优化；

- 后续作为离线数据采集器。


项目书明确规定 Online QMIX 和 Online MACC-QMIX 只是复现参考与数据采集器，不参与离线算法的公平排名。

## 先阅读四个源码位置

在动代码之前，把以下调用链梳理出来：

| 位置                                   | 需要弄清楚的内容                           |
| ------------------------------------ | ---------------------------------- |
| `src/modules/agents/macc_agent.py`   | 轨迹 RNN、原始子任务潜变量、注意力、个体 Q           |
| `src/controllers/macc_controller.py` | 输入构造、隐藏状态维护、agent 调用               |
| `src/learners/latent_q_learner.py`   | TD 目标、表示损失、target 更新和反向传播          |
| `src/run.py`                         | 环境交互、episode 收集、ReplayBuffer 和训练循环 |

重点画出两条数据流：

$$
\text{局部观测历史}
\rightarrow
\text{MAC（Multi-Agent Controller）}
\rightarrow
Q_i
\rightarrow
Q_{\mathrm{tot}}
$$

以及：

$$
\text{全局子任务状态}
\rightarrow
\text{表示监督损失}.
$$

## 第一阶段通过标准

不是一定要立即复现论文最终分数，而是满足：

- 连续训练不出现 NaN、CUDA 和维度错误；

- `td_loss`、重构损失和 Q 值日志正常变化；

- 测试回报不是永久固定；

- checkpoint 能保存、加载并复现评估结果；

- 能明确定位全局状态中每个子任务的字段；

- 能明确定位每个智能体局部观测中的实体字段；

- 能证明局部槽位和全局实体之间存在可靠对应。


### 第一个里程碑

$$
\boxed{
\begin{aligned}
&\text{LBF 上跑通 Online QMIX 和 Online MACC-QMIX}\\
+{}&\text{完成实体字段映射表}
\end{aligned}
}
$$

在完成这个里程碑之前，不要开始实现 OCSBC。

---

# 三、第二阶段：先建立离线数据管线

这是整个项目最容易被低估、但最关键的部分。

## 1. 把在线收集和离线训练彻底拆开

原始 PyMARL 流程通常是：

```python
episode_batch = runner.run(test_mode=False)
buffer.insert_episode_batch(episode_batch)
learner.train(...)
```

离线训练必须变成：

```python
dataset = OfflineEpisodeDataset(dataset_path)

for gradient_step in range(num_updates):
    batch = dataset.sample(batch_size)
    learner.train(batch, gradient_step)
```

训练期间必须保证：

```python
runner.run(test_mode=False)
```

完全不被调用。

环境只允许在定期评估时使用。

建议新增：

```text
src/collect_offline_dataset.py
src/components/offline_dataset.py
src/run_offline.py
src/envs/subtask_adapter.py
```

不要在原始 `run.py` 中堆积大量 `if offline:` 条件。实施计划也建议将数据采集、读取、训练入口和子任务接口解耦。

## 2. 扩展 EpisodeBatch 字段

除了原始字段，至少保存：

```python
scheme.update({
    "subtask_state": {
        "vshape": (K, subtask_state_dim)
    },
    "subtask_obs": {
        "vshape": (K, subtask_obs_dim),
        "group": "agents"
    },
    "subtask_visible": {
        "vshape": (K,),
        "group": "agents"
    },
    "subtask_mask": {
        "vshape": (K,)
    },
    "subtask_id": {
        "vshape": (K,)
    }
})
```

对应形状：

$$
\begin{aligned}
\text{subtask\_state} &: [B,T,K,d_u],\\
\text{subtask\_obs} &: [B,T,N,K,d_o],\\
\text{subtask\_visible} &: [B,T,N,K],\\
\text{subtask\_mask} &: [B,T,K].
\end{aligned}
$$

同时保留：

- 完整 episode 序列；

- `terminated`；

- `filled`；

- `avail_actions`；

- 全局状态；

- 局部观测；

- 数据动作；

- 奖励；

- 实体对应关系。


项目书要求所有离线方法使用完全相同的冻结数据，并保存完整序列、可用动作、全局状态、局部实体观测、可见性和实体对应关系。

## 3. 为环境定义统一子任务接口

建议封装：

```python
env.get_subtask_states()
env.get_agent_subtask_obs()
env.get_subtask_visibility()
env.get_subtask_mask()
env.get_subtask_ids()
```

第一版只支持 LBF 没有问题。等整个方法跑通后，再给 Predator-Prey 和 SMAC 实现相同接口。

## 4. 先构造小规模测试数据

不要一开始就收集大量 episode。分三步：

### 数据读写测试

每种行为质量先收集约 20～100 个 episode，用于：

- 保存和读取；

- shape 检查；

- mask 检查；

- 实体 ID 检查；

- 小数据过拟合。


### 初步实验

每种质量收集约 2,000～5,000 个 episode，具体数量再根据 episode 长度和学习曲线调整。

### 正式数据

根据初步结果确定规模，并固定数据版本和哈希。

## 5. 数据质量分组

项目书要求：

|类型|采集方式|
|---|---|
|低质量|随机策略或训练早期 checkpoint|
|中等质量|中期 checkpoint|
|高质量|收敛或近收敛 checkpoint|
|混合质量|多个 checkpoint 按比例合并|
|覆盖受限|删除特定实体交互或动作组合|

最开始只需做好 Medium 数据集，先验证整条离线管线。低、高、混合和覆盖受限数据在代码稳定后再补。

## 数据阶段通过标准

- 离线数据可反复加载，结果一致；

- 同一 episode 中实体槽位不漂移；

- padding、terminated 和有效时间步正确；

- 不可用动作保存正确；

- 训练期间环境交互次数严格为 0；

- 训练集和验证集按 episode 划分；

- 数据文件包含行为策略、checkpoint、epsilon、seed、环境版本和代码 commit；

- 一个很小的数据集能够被 Online/Offline 网络过拟合。


---

# 四、第三阶段：先做两个“直接离线化”基线

现在仍然不要改子任务表示。

## 1. Offline QMIX

保持：

- 原始 QMIX 网络；

- 原始 RNN；

- Double Q；

- target network；

- TD 损失；

- mixer。


只把数据来源改为固定数据。

目的不是追求好结果，而是回答：

> 普通 QMIX 直接使用固定数据时，会出现多大的性能下降和价值高估？

需要记录：

```text
td_loss
q_data_mean
q_max_mean
q_tot_abs_max
grad_norm
ood_action_rate
evaluation_return
```

## 2. Offline MACC-QMIX

保持原始 MACC：

- VAE 式或原始子任务表示；

- 原始重构损失；

- 原始注意力；

- 原始 QMIX mixer。


同样只把数据来源改为冻结数据。

该基线回答：

> 原始在线 MACC 的任务表示直接离线化是否已经足够？

如果 Offline QMIX 和 Offline MACC 都不能正常训练，优先检查：

- 时间序列错位；

- next-state 切片；

- terminated mask；

- target network；

- 数据加载；

- hidden state 重置。


此时不要用新信念模块“修补”离线骨干。

---

# 五、第四阶段：实现 CFCQL-QMIX

这是正式 OCSBC 的离线价值骨干。

## 正确的反事实计算

对于智能体 $i$：

1. 其他智能体使用数据动作；

2. 只替换智能体 $i$ 的动作；

3. 每个候选动作都重新经过 QMIX mixer；

4. 在反事实联合价值上计算保守正则。


$$
\mathcal R_{\mathrm{CF}}
=
\sum_i
\left[
\log\sum_{a_i}
\exp Q_{\mathrm{tot}}(\tau, a_i, \boldsymbol a_{-i}^{\mathcal D})
-
Q_{\mathrm{tot}}(\tau, \boldsymbol a^{\mathcal D})
\right].
$$

不能把它直接化简成：

$$
\log\sum_{a_i} \exp Q_i(a_i) - Q_i(a_i^{\mathcal D})
$$

因为这种化简只适用于 VDN 的线性求和，不适用于 QMIX。实施计划明确要求在 QMIX 输出的联合价值上计算保守正则。

## 建议独立模块

```text
src/learners/counterfactual_conservative.py
```

接口类似：

```python
cf_loss, cf_stats = compute_qmix_cf_loss(
    agent_q_all=mac_out,
    data_actions=actions,
    avail_actions=avail_actions,
    states=states,
    mixer=mixer,
    mask=mask,
)
```

## 必须记录

```text
cf_loss
q_data_mean
q_counterfactual_max_mean
q_gap
q_tot_abs_max
ood_action_rate
```

保守系数先搜索：

$$
\alpha \in \{0, 0.01, 0.1, 0.5, 1.0\}.
$$

## 同时建立 Conservative Offline MACC

将同一个 CFCQL-QMIX 保守实现接到原始 MACC 表示上：

$$
\text{原始 MACC 表示}
+
\text{原始注意力}
+
\text{反事实保守 QMIX}.
$$

这是 OCSBC 最重要的直接对照之一。因为只有比较：

$$
\text{Conservative Offline MACC}
\quad\text{vs.}\quad
\text{OCSBC}
$$

才能判断收益是否真的来自新的跨视角信念，而不只是来自保守正则。

### 这个阶段的通过标准

- `alpha=0` 时严格退化为普通 Offline QMIX；

- 不可用动作不进入 `logsumexp`；

- 数据动作路径与普通 TD 中完全一致；

- 保守损失数值稳定；

- 至少一个数据设置中，Q 值高估或 OOD 动作率有所改善；

- CFCQL-QMIX 与 Conservative Offline MACC 共享同一份保守正则实现。


---

# 六、第五阶段：逐步替换 MACC 的表示模块

只有前面的离线骨干稳定后，才开始实现本项目的主要创新。

不要一次实现完整 OCSBC，按以下顺序逐步加入。

## Step 1：确定性局部子任务信念

先去掉原 MACC 的高斯采样和标准正态 KL。

实现：

$$
h_i^t =
\operatorname{GRU}_{\tau}
(h_i^{t-1}, [o_i^t, a_i^{t-1}])
$$

$$
m_{ij}^{t} =
\operatorname{GRU}_{\mathrm{sub}}
(m_{ij}^{t-1}, [o_{ij}^t, v_{ij}^t])
$$

$$
z_{ij}^{t} =
E_{\mathrm{local}}([h_i^t, m_{ij}^t])
$$

关键要求：

- 所有实体共享同一个 `GRU_sub`；

- 不能给固定槽位使用不同参数；

- $z$ 的形状固定为 `[B,T,N,K,d_z]`；

- 执行路径只使用局部数据。


## Step 2：接入原有子任务注意力

$$
x_i^t =
\sum_j \alpha_{ij}^t W_v z_{ij}^t
$$

$$
Q_i = Q_i(h_i^t, x_i^t, a_i).
$$

此时先只使用 TD 和保守正则，检查新的确定性结构能否前向、反向并完成执行。

## Step 3：加入全局教师与重构

训练阶段：

$$
g_j^t = E_{\mathrm{global}}(u_j^{1:t}).
$$

局部信念重构停止梯度的教师表示：

$$
\mathcal L_{\mathrm{rec}}
=
\sum_{i,j}
w_{ij}^t
\left\|
D(z_{ij}^t) - \operatorname{sg}(g_j^t)
\right\|^2.
$$

这一版本就是：

$$
\text{Deterministic Belief w/o Contrastive}.
$$

它是后续证明“对比学习是否必要”的关键控制变量。

## Step 4：加入同场景跨视角对比

对于 $z_{ij}^t$：

- 正样本是 $g_j^t$；

- 负样本是同一时刻、同一 episode 中其他有效子任务 $g_k^t$。


$$
\mathcal L_{\mathrm{con}}
=
-\log
\frac{
\exp(\operatorname{sim}(P_l(z_{ij}), P_g(g_j))/T)
}{
\sum_k
\exp(\operatorname{sim}(P_l(z_{ij}), P_g(g_k))/T)
}.
$$

不要一开始把其他 episode 的实体全部当负样本。项目书选择同场景负样本，是为了减少地图、回合阶段和全局背景形成的捷径。

## Step 5：加入可见性时效权重

$$
w_{ij}^t =
\exp(-\Delta t_{ij}^t / \kappa).
$$

同时组合：

```python
belief_mask = (
    filled_mask
    * subtask_mask
    * visibility_weight
)
```

需要在编码前固定“从未观察过”的处理规则。建议设置为权重 0，避免对完全未知的实体状态施加强监督，但该规则应在所有实验版本中保持一致。

---

# 七、第六阶段：形成完整 OCSBC-QMIX

最终联合目标为：

$$
\boxed{
\mathcal L
=
\mathcal L_{\mathrm{TD}}
+
\alpha \mathcal R_{\mathrm{CF}}
+
\lambda_{\mathrm{rep}}
\left(
\mathcal L_{\mathrm{rec}}
+
\beta \mathcal L_{\mathrm{con}}
\right)
}
$$

完整方法采用单阶段、端到端训练：

- 不需要固定的两阶段预训练；

- 不长期冻结信念编码器；

- 训练时使用全局教师和 mixer；

- 执行时只保留局部轨迹编码、实体记忆、局部信念、注意力和个体 Q。


执行阶段必须能够在不读取以下信息的情况下完成动作选择：

```text
global state
subtask_state
global teacher
decoder
contrastive projector
QMIX mixer
CF regularizer
```

---

# 八、最初不要铺开所有环境和基线

## 开发阶段只做一个环境

建议：

$$
\boxed{\text{LBF + 一个固定任务配置 + Medium 数据集}}
$$

原因是当前目标不是证明最终性能，而是验证：

- 实体对应是否正确；

- 信念记忆是否正确；

- 可见性权重是否正确；

- 同场景对比标签是否正确；

- 离线训练是否完全无交互。


## 第一轮最小实验矩阵

|编号|方法|主要验证|
|---|---|---|
|B1|Online QMIX|环境和 QMIX|
|B2|Online MACC-QMIX|原始 MACC 复现|
|B3|Offline QMIX|离线数据与 TD|
|B4|Offline MACC-QMIX|原表示直接离线化|
|B5|CFCQL-QMIX|保守价值骨干|
|B6|Conservative Offline MACC|相同保守骨干下的原始表示|
|A1|Deterministic Belief w/o Contrastive|确定性信念与重构|
|Full|OCSBC-QMIX|完整方法|

暂时不要做：

- OMIGA；

- ComaDICE；

- InSPO；

- SMACv2；

- 大规模超参数搜索；

- VDN 正式实验；

- 所有消融同时展开。


主结果稳定后再加入 OMIGA，其他方法属于资源允许时的扩展。

---

# 九、建议你接下来七天这样开始

## 第 1 天：环境和仓库检查

- 创建独立环境；

- 跑通原始仓库自带示例；

- 建立代码版本记录；

- 创建 `reproduce-qmix` 和 `reproduce-macc-qmix` 分支；

- 记录 Python、PyTorch、CUDA、环境版本。


## 第 2 天：Online QMIX smoke test

- 选择 LBF；

- 跑小规模训练；

- 检查 RNN、target network、mixer 和评估；

- 确认 checkpoint 保存、加载。


## 第 3 天：Online MACC-QMIX smoke test

- 记录原始表示损失；

- 记录注意力输出；

- 找到局部实体字段和全局子任务字段；

- 画出 MACC 的完整源码调用图。


## 第 4 天：实现子任务 adapter

实现：

```python
get_subtask_states()
get_agent_subtask_obs()
get_subtask_visibility()
get_subtask_mask()
get_subtask_ids()
```

并用人工构造的小 episode 检查实体 ID 一致性。

## 第 5 天：实现离线数据采集与保存

- 从一个固定 MACC checkpoint 收集约 100 个 episode；

- 保存完整序列和 metadata；

- 写数据完整性检查脚本。


## 第 6 天：实现离线数据读取

- 实现 `OfflineEpisodeDataset`；

- 比较保存前后 batch 是否逐元素一致；

- 检查 padding、terminated、avail actions 和实体 mask；

- 确认训练循环不再收集新 episode。


## 第 7 天：跑通 Offline QMIX 小数据测试

- 不加入保守正则；

- 不加入子任务信念；

- 尝试在极小数据集上过拟合；

- 记录 TD、Q 值和评估回报；

- 检查训练期间环境交互次数为 0。


### 第一周结束时的验收结果

你应该拥有：

$$
\boxed{
\begin{aligned}
&\text{Online QMIX 可运行}\\
+{}&\text{Online MACC-QMIX 可运行}\\
+{}&\text{LBF 子任务接口}\\
+{}&\text{100 个 episode 的冻结数据}\\
+{}&\text{Offline QMIX 可训练}.
\end{aligned}
}
$$

第一周不需要完成对比学习。此时最大的成功不是获得高回报，而是证明：

> **原始代码、实体对应、离线数据和固定数据训练都正确。**

完成这一里程碑后，再开始 CFCQL-QMIX；保守骨干通过验收后，才进入子任务信念模块。这样最符合项目书的变量控制原则，也最容易形成后续可以写进论文的完整实验归因链。
