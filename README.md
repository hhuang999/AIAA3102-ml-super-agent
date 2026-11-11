# AIAA3102 Machine Learning SuperAgent

AIAA3102 Final Project 工作区。目标是开发一个面向表格、文本和图像分类任务的通用 Agent，自主分析数据、提出方案、执行实验、依据验证结果改进，并交付可运行的预测方案。

当前阶段：**学校 HPC 基础训练环境已配置，CUDA 与小规模训练验证通过；等待开发 SPEC，尚未实现 SuperAgent。**

## 项目材料

- [课程要求](<docs/AIAA3102 - Final Project - Full Assignment Requirements (1).md>)：课程约束与交付标准。
- [环境与开发准备记录](docs/workspace-readiness.md)：本机、server、HPC、API 的实测结果与开发建议。
- [Group 与学校 HPC 对比](docs/hpc-comparison.md)：更新连接资料后的正确登录结果、资源、申请测试与环境建议；远程环境以此为准。
- [学校 HPC 训练环境](docs/hpc-training.md)：已验证的环境路径、正常排队命令、GPU UUID 绑定与实测结果。
- `final_project_tasks/`：课程提供的两个任务、原始数据和任务说明；当前数据总量约 3 MB，随 private 仓库保存。
- `scripts/check_environment.py`：本机环境检查入口。
- `AGENTS.md`：后续开发 Agent 的项目入口与既有决定。

收到 SPEC 后放入 `docs/`，再确定 Agent 架构、依赖和统一执行入口。基础训练环境的依赖输入为 `requirements-hpc.txt`，已验证的完整版本为 `requirements-hpc.lock.txt`；安装与训练检查入口见 `scripts/hpc/prepare_and_test.sbatch`。

## 已验证的本机环境

Windows / Conda `vb` / Python 3.12.11 / RTX 4060 Laptop 8GB。PowerShell 下直接使用解释器，避免默认 `python` 指向另一个安装：

```powershell
& 'D:\anaconda\envs\vb\python.exe' scripts/check_environment.py
& 'D:\anaconda\envs\vb\python.exe' scripts/check_environment.py --check-cuda --check-data
```

脚本只检查 train/val 的结构，不读取测试集内容。`--check-cuda` 会执行一次小型 GPU 运算。后续依据 SPEC 声明项目实际依赖；当前环境不是最终可复现的依赖锁定文件。

本机 `conda.exe` 在当前 shell 中出现启动错误，但以下入口已可用：

```powershell
& 'D:\anaconda\python.exe' -m conda env list
```

## 本地配置与 Git

私有远程仓库：[hhuang999/aiaa3102-ml-super-agent](https://github.com/hhuang999/aiaa3102-ml-super-agent)。使用已有 GitHub SSH 身份同步；GitHub CLI 的现有 token 未通过鉴权。

`.env.example` 仅说明可配置的变量。现有 `.env`、两份连接凭据及 `docs/course-api-access.md` 保留在本机并由 `.gitignore` 排除。运行产物、模型权重和临时检查文件也不进入 Git。Git clone 后需要单独配置凭据。

## 开发必须保留的课程约束

- 相同最终版本处理所有任务，以数据特点与实验结果决策，不以任务名称分发人工预制方案。
- 正式结果来自完整自主运行；运行中由 Agent 决定实验和最终模型。
- 测试集仅供最终评估，训练、调参和模型选择使用训练集与验证集。
- 每项评估任务的全部 LLM 调用合计预算不超过人民币 20 元。按 2026-10-21 官方标准价格计费，保存各 Agent 的输入/输出 token 与成本记录。
- 仅使用在 2026-10-21 当日或之前正式发布、具备官方标准 API 定价的 Qwen / DeepSeek 模型。API 可调用不代表已经核实课程适用性。

详细要求以课程原文及后续 SPEC 为准。
