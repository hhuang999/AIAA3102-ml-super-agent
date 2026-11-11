# Group 与学校 HPC 对比

检查日期：2026-11-11，Asia/Singapore。GPU 占用与共享存储容量为检查时快照。

**后续进展：已决定优先使用学校 HPC，并完成实际 GPU 训练验证。** 成功作业 `13847157` 在 A40 上完成 CUDA、ResNet18 反向传播、10 epochs 合成分类训练和 checkpoint 重载，退出码为 0。环境已保存在 SSD，GPU 已释放。配置、关键修正及运行入口见 [学校 HPC 训练环境](hpc-training.md)。下文保留资源选型时的检查记录。

## 资料修正

用户更新了 `2.server_hpc.txt`：当前学校 HPC 入口是 `https://login.hpc.hkust-gz.edu.cn/appform/login`，使用该文件中的课程账号。此前外部共享单卡 A6000 server 的检查结果不再代表当前候选环境。

`1.group_HPC.txt` 在同一行提供账号、密码及 `；ip:` 字段。此前错误地把 IP 字段拼入密码，并将该账号用于学校入口，所得认证失败不能用于判断 group HPC 是否可用。本次正确解析字段后，group SSH 登录成功。

## 已确认的比较

| 维度 | Group HPC | 学校 HPC |
| --- | --- | --- |
| 入口 | 本地 `1.group_HPC.txt`，SSH，主机 `star` | `2.server_hpc.txt`，门户和 SSH，登录节点 `mgmt-3` |
| 登录 | SSH 成功 | ego-browser 门户登录成功，SSH 登录成功 |
| GPU | 8 × RTX A6000 48GB | Slurm 列出 A800 80GB、A40 48GB；另有独立 AI 容器平台 |
| 当前 GPU 情况 | 8 卡利用率 98%–100%，剩余显存约 0.07–12GB | Slurm 查询时部分 A800 节点尚有未分配 GPU；实际可启动性需由调度验证 |
| CPU / RAM | 当前 shell 可见 224 CPU、约 1TiB 内存，约 782GiB available；共享资源 | 按作业申请 CPU、内存与 GPU；登录节点资源不用于训练 |
| Home 存储 | `/home` 总计 7T，100% 已用，仅余约 624MB | `/hpc2hdd` 共享文件系统约 86T 可用，使用率 98% |
| 其他存储 | `/Data` 可写，共享文件系统约 31T 可用 | `/hpc2ssd` 约 94T 可用，使用率 70%；账号工作目录可写 |
| 网络 | PyPI 和校内 AIGC API 返回 HTTP 200；Hugging Face 连接被重置 | 登录节点上 PyPI 和校内 AIGC API 返回 HTTP 200；Hugging Face 连接被重置 |
| 默认 Python | 3.10.12；系统解释器未发现 PyTorch、pandas、sklearn 等 | 登录节点 Python 3.8.10；这不是后续训练环境版本 |
| 调度 | 本次未发现 Slurm 命令；现有 GPU 由共享进程占用 | Slurm 可查询分区、账号 association、节点与个人作业 |

存储剩余量是整个文件系统的剩余容量，不是个人配额。两边个人配额尚未核实。网络检查未使用 API key，不增加模型调用。

Group 账号现有 home 数据约 53GB，包含先前项目。本次仅查看目录容量与环境位置，未修改。已有 `aihospital/.venv`，尚未核实其中的 PyTorch/CUDA 适配性。后续若使用 group，应将本项目环境、缓存与结果放到 `/Data` 下专属目录，并限制 CPU 线程；不向已满的 home 安装大型环境。

## 学校门户申请与 SSH 验证

按用户要求，先通过 ego-browser 登录、查看开发环境与资源规格，再申请测试环境：

- 名称：`aiaa3102-ml-preflight-1111`
- 平台 / 作业号：`AI-X86_NVIDIA` / `99261`
- 镜像：`admin/webide-jupyter:v6.2`
- 资源：4 CPU、A800 MIG 10GB、2GB shared memory，启用 SSH
- 结果：在本轮观察窗口内持续显示“启动中”，未取得可验证的容器 SSH 入口；随后通过门户停止，刷新确认状态“退出”。没有修改其他已有环境。

门户中整卡资源规格的可分配节点数量显示为 0；MIG 规格没有显示明确数量。这个 AI 容器平台与 Slurm 是不同资源池，不能据此判断学校全部 GPU 都不可用。

登录节点 SSH 成功，home 与工作目录可写。需要先执行：

```bash
source /opt/slurm/profile.slurm
sinfo
squeue -u "$USER"
```

未加载该 profile 时，PATH 中 `/usr/bin/sinfo` 使用了错误的默认配置位置，报缺少 `/etc/slurm-llnl/slurm.conf`；`/opt/slurm/bin` 下的版本查询正常。已查到账号 association 允许 debug、A800、A40 等 QOS。

一次 `debug` 队列 A40 单卡短作业（`13846459`）在 20 秒申请窗口内未能分配资源，返回 `Requested nodes are busy`；检查后个人作业列表为空。这一结果仅说明该请求当时未立即获得资源。

随后检查 A800 节点的 `CfgTRES` 与 `AllocTRES`，发现部分节点存在未分配 GPU。按账号允许的 `i64m1tga800u` 分区与同名 QOS 申请 1 张 A800、8 CPU、16GB RAM、2 分钟运行时限（`13846505`），在 60 秒分配等待窗口内同样返回 `Requested nodes are busy`。这说明表面未分配的 GPU 并不保证当前请求能立即获得资源，排队优先级等条件仍由调度器决定。两次短作业均已结束，确认个人 Slurm 作业列表为空；没有执行 GPU 训练或 CUDA 运算。

门户个人余额显示 `-1.77 元`。历史账单中有收费记录，也有费用为 0 的记录；当前课程账号的收费政策与余额对作业的影响仍待核实。本次没有充值或修改账户设置。

## 初步建议

优先采用的方向是**本机开发与 Agent 控制 + 学校 Slurm GPU 作业**。学校调度器能明确分配资源，并提供较大显存与持久共享存储，适合记录环境、资源和运行日志。当前两处都没有验证出可立即使用的独占 GPU；学校仍需通过正常排队获得实际分配，并完成训练栈 smoke test，才能认定已具备正式训练条件。

Group 可作为 CPU 开发与 GPU 备选：CPU/内存和 `/Data` 容量充足，但当前 8 张卡都在高负载，不能把剩余显存视为可独享的训练资源。

正式依赖环境在 SPEC 后配置；统一 Python 版本和项目依赖，明确模型缓存位置、外网权重同步、作业时限、任务结果回传及费用记录。两边都需要为 Hugging Face 下载准备可用路径。
