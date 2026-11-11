# 项目入口

## 当前阶段

这是 AIAA3102 Final Project 的 Machine Learning SuperAgent 项目。2026-11-11 完成工作区和资源检查；等待用户提供 SPEC，再开展 Agent 实现。当前没有 Agent、训练入口或正式实验结果。

先读 `README.md`、`docs/workspace-readiness.md`、`docs/hpc-comparison.md` 和 `docs/AIAA3102 - Final Project - Full Assignment Requirements (1).md`。远程资源以更新后的 HPC 对比记录为准。新 SPEC 放在 `docs/`，据此更新实施安排。

## 已有材料与环境

- `final_project_tasks/` 保存原始课程材料。保留目录与文件，不覆盖原始数据。
- 本机默认使用 `D:\anaconda\envs\vb\python.exe`，不是 PATH 上的系统 Python。环境检查：`scripts/check_environment.py --check-cuda --check-data`。
- `D:\anaconda\python.exe -m conda` 可用；直接运行 `conda.exe` 曾出现启动错误。
- 本机 RTX 4060 Laptop 8GB 已通过 CUDA 运算检查。
- `1.group_HPC.txt` 同一行含账号、密码及 `；ip:` 字段，解析密码时必须截断 IP 字段。正确入口的 SSH 已通过；主机 `star` 有 8 张 A6000，检查时全部高负载。`/home` 已满，`/Data` 可写且剩余约 31T，尚未配置本项目环境。
- `2.server_hpc.txt` 已更新为学校 HPC 门户账号，门户与 SSH 均登录成功。先 `source /opt/slurm/profile.slurm`，再使用 Slurm 命令；PATH 默认发行版命令会误报配置缺失。学校有 A800/A40 资源与大容量共享存储；本轮 A40/A800 短时分配均因 busy 未获得 GPU，作业列表已清空。后续需要正常排队、验证训练栈并确认费用政策。
- 本轮通过门户创建的 MIG 测试环境 `aiaa3102-ml-preflight-1111`（99261）已停止；其他已有环境保持原状。此前外部共享单卡 server 和错误解析凭据所得的认证结论已过时。
- 课程 API 的模型列表和最小 Qwen 调用通过。模型发布时间及官方历史价格仍待核实，正式选型依 SPEC 进行。

## Git 与配置

private 仓库：`git@github.com:hhuang999/aiaa3102-ml-super-agent.git`。默认分支 `main`；具体开发使用 `codex/` 分支。

本地 `.env`、`1.group_HPC.txt`、`2.server_hpc.txt`、`docs/course-api-access.md` 含敏感配置，由 `.gitignore` 排除。读取时避免将值打印至日志。新增配置只在 `.env.example` 中保存变量名和公开默认值。

## 实施重点

将课程自主运行、测试集隔离、全部 Agent 的 token/成本记账、人民币 20 元任务预算、统一最终版本与可复现预测产物纳入架构与验收。模型标识与价格须冻结到课程允许的日期和版本。

当前环境方案是待 SPEC 确认的建议：本机控制与 API 调用，优先评估学校 Slurm GPU worker，group 作为 CPU/GPU 备选。还没有部署或实现 Agent 执行方式。
