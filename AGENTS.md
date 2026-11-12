# 项目入口

## 当前阶段

这是 AIAA3102 Final Project 的 Machine Learning SuperAgent 项目。2026-11-11 已完成学校 HPC 基础训练环境配置与实际 CUDA / 小规模训练验证；等待用户提供 SPEC，再开展 Agent 实现。当前没有 SuperAgent 或课程正式实验结果。

先读 `README.md`、`docs/hpc-training.md`、`docs/hpc-comparison.md` 和 `docs/AIAA3102 - Final Project - Full Assignment Requirements (1).md`。远程环境的实际配置与运行方式以训练环境记录为准；`docs/workspace-readiness.md` 保留首次检查历史。新 SPEC 放在 `docs/`，据此更新实施安排。

## 已有材料与环境

- `final_project_tasks/` 保存原始课程材料。保留目录与文件，不覆盖原始数据。
- 本机默认使用 `D:\anaconda\envs\vb\python.exe`，不是 PATH 上的系统 Python。环境检查：`scripts/check_environment.py --check-cuda --check-data`。
- `D:\anaconda\python.exe -m conda` 可用；直接运行 `conda.exe` 曾出现启动错误。
- 本机 RTX 4060 Laptop 8GB 已通过 CUDA 运算检查；`vb` 中 torchvision 0.23.0 与 torch 2.6.0 的组合导入失败，图像训练优先使用已验证的 HPC 环境。
- `1.group_HPC.txt` 同一行含账号、密码及 `；ip:` 字段，解析密码时必须截断 IP 字段。正确入口的 SSH 已通过；主机 `star` 有 8 张 A6000，检查时全部高负载。`/home` 已满，`/Data` 可写且剩余约 31T，尚未配置本项目环境。
- 学校 HPC 已确定为首选；`2.server_hpc.txt` 中的门户、SSH 账号可用。先 `source /opt/slurm/profile.slurm`，再用 Slurm 命令；应在 `set -u` 之前加载该 profile。PATH 默认发行版命令会误报配置缺失。
- 已验证的解释器为 `/hpc2ssd/JH_DATA/spooler/dsaa2012_006/aiaa3102-super-agent/.venv/bin/python`：Python 3.11.9、PyTorch 2.6.0、CUDA runtime 12.4。完整依赖在 `requirements-hpc.lock.txt`，优先用校内 Nexus 镜像。
- GPU 作业中若 NVML 仅暴露一张卡，将 `CUDA_VISIBLE_DEVICES` 设为该卡 UUID；本次 Slurm 节点级数字编号与进程内编号不一致，导致未经此处理的 CUDA 初始化失败。复用 `scripts/hpc/prepare_and_test.sbatch` 的绑定逻辑，保留资源分配边界。
- 成功验证作业 `13847157` 在 A40 上以 `COMPLETED / 0:0` 结束，矩阵计算、ResNet18 反向、10 epochs 合成分类训练、checkpoint 重载和训练库导入全部通过。GPU 已释放，环境与结果保留；见 `docs/hpc-validation-result.json`。
- 环境准备默认 `debug` 分区及同名 QOS，最多 30 分钟。长作业应同时指定匹配的常规 partition/QOS；常规队列可能等待较久，个人存储配额与费用政策尚未完全核实。
- 2026-11-12 只读校验确认学校提交规则为每张 GPU 最多8 CPU；1 GPU / 16 CPU会被拒绝。MLEvolve原配置的21 CPU须按实际Slurm分配调整。debug只供代码调试；正式Agent建议在一次常规GPU作业内连续运行，资源、费用及队列预测见 `docs/resources-for-ml-agent.md`。
- 本轮通过门户创建的 MIG 测试环境 `aiaa3102-ml-preflight-1111`（99261）已停止；其他已有环境保持原状。此前外部共享单卡 server 和错误解析凭据所得的认证结论已过时。
- 课程 API 的模型列表和最小 Qwen 调用通过。模型发布时间及官方历史价格仍待核实，正式选型依 SPEC 进行。

## Git 与配置

private 仓库：`git@github.com:hhuang999/aiaa3102-ml-super-agent.git`。默认分支 `main`；具体开发使用 `codex/` 分支。

本地 `.env`、`1.group_HPC.txt`、`2.server_hpc.txt`、`docs/course-api-access.md` 含敏感配置，由 `.gitignore` 排除。读取时避免将值打印至日志。新增配置只在 `.env.example` 中保存变量名和公开默认值。

## 实施重点

将课程自主运行、测试集隔离、全部 Agent 的 token/成本记账、人民币 20 元任务预算、统一最终版本与可复现预测产物纳入架构与验收。模型标识与价格须冻结到课程允许的日期和版本。

已决定优先使用学校 HPC；当前采用本机开发与控制、学校 Slurm GPU 执行的方向，group 作为备选。基础环境已就绪，Agent 的执行架构与正式依赖仍需依据 SPEC 实现。
