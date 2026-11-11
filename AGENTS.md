# 项目入口

## 当前阶段

这是 AIAA3102 Final Project 的 Machine Learning SuperAgent 项目。2026-11-11 完成工作区和资源检查；等待用户提供 SPEC，再开展 Agent 实现。当前没有 Agent、训练入口或正式实验结果。

先读 `README.md`、`docs/workspace-readiness.md` 和 `docs/AIAA3102 - Final Project - Full Assignment Requirements (1).md`。新 SPEC 放在 `docs/`，据此更新实施安排。

## 已有材料与环境

- `final_project_tasks/` 保存原始课程材料。保留目录与文件，不覆盖原始数据。
- 本机默认使用 `D:\anaconda\envs\vb\python.exe`，不是 PATH 上的系统 Python。环境检查：`scripts/check_environment.py --check-cuda --check-data`。
- `D:\anaconda\python.exe -m conda` 可用；直接运行 `conda.exe` 曾出现启动错误。
- 本机 RTX 4060 Laptop 8GB 已通过 CUDA 运算检查。
- 共享 server 的 A6000 48GB 可见且 SSH 登录成功，但 home 磁盘接近满、无法访问 Hugging Face、不能解析校内 API 域名。尚未准备训练环境。使用前重新检查容量和 GPU 占用，保留其他学生的任务。
- HPC 一、二、三期可建立 SSH 连接，现有凭据均被拒绝；配额、分区和调度权限未验证。
- 课程 API 的模型列表和最小 Qwen 调用通过。模型发布时间及官方历史价格仍待核实，正式选型依 SPEC 进行。

## Git 与配置

private 仓库：`git@github.com:hhuang999/aiaa3102-ml-super-agent.git`。默认分支 `main`；具体开发使用 `codex/` 分支。

本地 `.env`、`1.group_HPC.txt`、`2.server_hpc.txt`、`docs/course-api-access.md` 含敏感配置，由 `.gitignore` 排除。读取时避免将值打印至日志。新增配置只在 `.env.example` 中保存变量名和公开默认值。

## 实施重点

将课程自主运行、测试集隔离、全部 Agent 的 token/成本记账、人民币 20 元任务预算、统一最终版本与可复现预测产物纳入架构与验收。模型标识与价格须冻结到课程允许的日期和版本。

当前环境方案是待 SPEC 确认的建议：本机控制与 API 调用，必要时经 SSH 使用 Linux GPU worker。还没有部署或实现该执行方式。
