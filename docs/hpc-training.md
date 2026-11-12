# 学校 HPC 训练环境

**已配置并通过实际 GPU 训练验证。** 环境位于账号的 SSD 工作目录，保留供后续开发使用；验证作业已经结束并释放 GPU。

## 路径与配置

项目根目录：`/hpc2ssd/JH_DATA/spooler/dsaa2012_006/aiaa3102-super-agent`。

| 内容 | 相对项目根目录的路径 |
| --- | --- |
| Python 虚拟环境 | `.venv/` |
| Python 依赖输入 | `requirements-hpc.txt` |
| 已验证的完整依赖锁定 | `requirements-hpc.lock.txt`（62 个包） |
| 安装与测试批作业 | `scripts/hpc/prepare_and_test.sbatch` |
| CUDA / 训练验证脚本 | `scripts/hpc/smoke_train.py` |
| pip / 临时下载缓存 | `cache/pip/`、`cache/tmp/` |
| 模型缓存 | `cache/huggingface/` |
| 作业日志 | `logs/aiaa3102-env-check-<job_id>.log` |
| 测试结果 | `runs/hpc-smoke/<job_id>/` |

解释器基于学校 `anaconda3` module 提供的 Python 3.11.9 创建隔离 venv。`requirements-hpc.txt` 声明基线训练依赖；正式 SuperAgent 的依赖在 SPEC 后补充。PyTorch 2.6.0 / torchvision 0.21.0 的版本组合参照[官方安装说明](https://docs.pytorch.org/get-started/previous-versions/)。所选官方 Linux PyPI `torch==2.6.0` wheel 的发布元数据声明 CUDA runtime 12.4.127、cuDNN 9.1.0.70。

依赖使用学校 Nexus PyPI 镜像 `https://nexus.hpc.hkust-gz.edu.cn/repository/pypi-hkust/simple`。按[学校镜像文档](https://docs.hpc.hkust-gz.edu.cn/docs/hpc3/on-campus-repository/)核对仓库路径，并实测 HTTPS 入口可用；cuDNN wheel 的 8MB 片段传输约 35MB/s。直接外网源下载明显较慢，因此首次配置中切换到校内镜像。只修改项目依赖输入，不修改全局 pip 配置。

本机 `vb` 的 PyTorch 为 2.6.0+cu124、torchvision 为 0.23.0，本轮本机尝试导入 torchvision 报 `operator torchvision::nms does not exist`。远程使用独立环境和上述官方配套版本；本轮没有修改本机 `vb`。

## 提交、检查与重跑

先用本地连接资料登录学校 SSH，再执行：

```bash
source /opt/slurm/profile.slurm
cd /hpc2ssd/JH_DATA/spooler/dsaa2012_006/aiaa3102-super-agent
mkdir -p logs
sbatch --parsable --output="$PWD/logs/%x-%j.log" scripts/hpc/prepare_and_test.sbatch
squeue -u "$USER"
```

批作业默认在 `debug` 分区正常排队，申请一张 GPU、8 CPU、32GB RAM，最多运行 30 分钟。该时限从分配后开始计时，不是排队超时。安装、验证结束后批脚本退出并释放资源。后续单纯重跑验证时，已安装依赖会复用，可用 `sbatch --time=00:05:00 ...` 缩短运行时限。

学校将 debug 定位为代码调试免费试用。正式 ML Agent 长跑应选择常规队列，避免将正式实验拆成连续 debug 作业。2026-11-12 实测提交规则还要求每张 GPU 最多8 CPU；进一步的资源与费用比较见 [ML Agent 资源方案](resources-for-ml-agent.md)。

需要常规 A800 队列时，在同一 `sbatch` 命令中同时覆盖 `--partition=i64m1tga800u --qos=i64m1tga800u`；A40 常规队列对应 `i64m1tga40u`。学校提交入口会检查 partition/QOS 组合，直接提交公共 QOS `gpu16` 曾被拒绝，因此默认脚本采用匹配的 `debug` 组合。

`/opt/slurm/profile.slurm` 引用可能尚未设置的环境变量，应先加载，再开启 shell 的 `set -u`。批脚本本身不需要重新加载该 Slurm profile。

使用已配置的解释器运行训练仍应在 GPU 作业内进行，并复用批脚本的 GPU UUID 绑定。该环境曾出现 Slurm 提供节点级编号、进程内 NVML 重新从 0 编号的情况；仅在 NVML 暴露一张 GPU 时，将 mask 转为这张已分配卡的 UUID：

```bash
mapfile -t allocated_gpu_uuids < <(nvidia-smi --query-gpu=uuid --format=csv,noheader)
if [[ ${#allocated_gpu_uuids[@]} -eq 1 ]]; then
    export CUDA_VISIBLE_DEVICES="${allocated_gpu_uuids[0]}"
fi
/hpc2ssd/JH_DATA/spooler/dsaa2012_006/aiaa3102-super-agent/.venv/bin/python your_training_script.py
```

该处理仅把已分配卡的标识从编号转换为 UUID。批脚本保存原始编号到 `SCHEDULER_CUDA_VISIBLE_DEVICES` 以便核对。参考 [Slurm GRES 设备编号说明](https://slurm.schedmd.com/gres.html)、[NVIDIA CUDA_VISIBLE_DEVICES UUID 说明](https://docs.nvidia.com/deploy/topics/topic_5_2_1.html)。

## 验证标准

测试仅使用合成数据和随机初始化模型，不读取课程任务数据，也不调用 LLM：

1. PyTorch 只看到分配的单张 GPU，CUDA 可用。
2. CUDA 矩阵乘法与 CPU 参考结果一致。
3. 随机初始化 ResNet18 完成前向和反向传播，loss 与全部梯度有限。
4. 生成 10,000 条二分类样本，按 8,000 / 2,000 划分训练与验证集，仅用训练集拟合标准化参数。
5. MLP 在 GPU 上训练 10 epochs，最终训练 loss 低于首轮的 60%，验证 accuracy 不低于 90%。
6. 模型与预处理参数保存到 checkpoint；重载模型后，验证 logits 完全一致。
7. PyTorch、torchvision、pandas、sklearn、CatBoost、LightGBM、XGBoost、Transformers 可用，并保存实际 `pip freeze`。

输出 `metrics.json`、`classifier.pt`、`pip-freeze.txt`，日志末尾应包含 `HPC_VALIDATION_PASSED`。这些测试验证训练环境，不代表 SuperAgent 或课程任务模型已经实现。

## 本轮实测结果

成功作业：**`13847157`，`debug` 分区，`gpu3-9` 节点，`COMPLETED`，退出码 `0:0`**。2026-11-11 23:37:57（Asia/Singapore）完成验证，23:37:59 批脚本正常结束。该次作业复用已安装依赖，总运行 15 秒；GPU 已释放，账号当前没有本任务的活动作业。

| 检查 | 结果 |
| --- | --- |
| GPU | NVIDIA A40，标称 48GB，PyTorch 可见约 47.40GiB，单卡 |
| 驱动 / CUDA runtime / cuDNN | 575.57.08 / 12.4 / 9.1.0 |
| Python / PyTorch / torchvision | 3.11.9 / 2.6.0 / 0.21.0 |
| CUDA 矩阵运算 | 与 CPU 参考结果一致 |
| ResNet18 | GPU 前向、反向及有限梯度检查通过 |
| 合成分类训练 | 8,000 训练样本、2,000 验证样本，10 epochs |
| 训练 loss | 首轮 0.180691 → 最后一轮 0.022856 |
| 最终验证 accuracy | 99.10% |
| MLP 训练时间 | 约 1.68 秒，不含安装与其他检查 |
| Checkpoint | 保存、重载后的验证 logits 完全一致 |
| 训练库 / 依赖 | 全部导入通过，`pip check` 通过 |
| 存储 | 独立 `.venv` 约 6.4GB，缓存约 3.3GB；下载临时文件已清理 |

完整数值见 [验证结果 JSON](hpc-validation-result.json)。原始日志、checkpoint、`pip freeze` 已下载到本机 `runs/hpc-smoke/13847157/`，SHA256 与远程文件一致；远程同样保留原件。锁文件已通过现有环境的 pip dry-run 和依赖检查，批脚本后续优先使用锁文件。

首次正常排队作业为 `13846817`，23:13:52 提交。常规队列曾预估到 11 月 15 日，扩展到短作业池后于 23:22 获得 A40。外网下载较慢，保留 ID 重新入队一次并切换校内镜像；依赖安装成功，但 CUDA 初始化失败。当时 Slurm 导出编号 `7`，NVML 只暴露编号 `0`，PyTorch 没有发现 GPU。后续作业改为已分配 GPU 的 UUID 绑定，全部测试通过。初次失败及切换源前的日志保留在远程 `logs/`。

本次完成的基线足以开展后续代码开发与小规模训练。正式任务仍按所需时长选择队列；短作业池上限为 30 分钟，常规 A800/A40 队列等待时间可能更长。

学校共享存储容量不是个人配额；账号费用以平台实际账单为准。Hugging Face 访问受限的问题独立于本次训练验证，后续模型权重仍需准备可用下载／同步路径。
