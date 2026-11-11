# 工作区与环境检查

检查日期：2026-11-11（Asia/Singapore）。以下是当前会话的实测快照；共享资源占用和容量会变化。

**连接资料已更新。** 本文旧 server 与 HPC 认证部分属于首次检查历史。当前 group 与学校 HPC 的正确入口、登录结果和建议见 [HPC 对比记录](hpc-comparison.md)，后续环境决策以该记录为准。

## 工作区状态

实际路径：`D:\HHB\HKUST(GZ)_Library_code\AIAA3102_Python_for_AI\Final_project_agent`。当前工作区包含课程要求、API 接入说明、server/HPC 凭据和两个课程任务，没有已有 Agent 实现或项目依赖声明。后续开发 SPEC 尚未提供。

课程目标是同一套通用 Agent 自主处理表格、文本、图像分类，完成实验与最终预测交付。自主性与泛化设计占 30%，已发布任务性能占 20%，未见任务性能占 10%；需要兼顾完整闭环、可复现性与性能，不能只优化两个已知任务。

### 数据材料

| 任务 | Train | Validation | Test | 输入 / 指标 |
| --- | ---: | ---: | ---: | --- |
| Coupon recommendation | 8,117 | 2,030 | 2,537 | 21 列类别特征，二分类 / accuracy |
| Headlines CTR | 8,301 | 940 | 3,232 | 成对 headline 文本，二分类 / accuracy |

Train/validation 的行数、数组形状和 CSV 列名已读取核对。Test 数量来自任务说明与 `info.json`，没有加载测试内容或进行模型评估。当前没有训练、基线成绩或正式实验结果。

## 本机：可直接用于开发

| 项目 | 实测结果 |
| --- | --- |
| 系统 | Windows 11 |
| 推荐解释器 | `D:\anaconda\envs\vb\python.exe` |
| Python | 3.12.11 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，8GB |
| PyTorch | 2.6.0+cu124，CUDA 12.4 |
| CUDA 验证 | 可用，GPU 张量平方求和得到 5.0 |
| NumPy / pandas / scikit-learn | 2.3.5 / 2.3.3 / 1.7.1 |
| CatBoost / LightGBM / XGBoost | 1.2.10 / 4.6.0 / 3.2.0 |
| Transformers | 4.57.6 |
| OpenAI SDK / Pydantic | 2.44.0 / 2.13.4 |
| Paramiko / httpx | 5.0.0 / 0.28.1 |

包版本来自已安装元数据；并未逐一执行所有库的训练功能。PATH 默认 Python 指向另一份系统 Python 3.12，后续命令使用完整解释器路径。`conda.exe` 及 shell profile 中的 Conda hook 出现启动错误；`D:\anaconda\python.exe -m conda` 成功返回环境信息。当前检查用 `login=false` 跳过 profile，未修改用户全局配置。

本轮复用 `vb`，没有安装、升级或删除依赖。SPEC 确认后，依据实际实现生成依赖声明与干净环境复现流程；不把整个通用环境直接作为项目 requirements。

## 旧外部共享 server：已被用户替换

连接信息保存在本地 `2.server_hpc.txt`。密码未进入受版本管理的材料。

| 项目 | 实测结果 |
| --- | --- |
| SSH | 账号登录成功，可执行命令，home 可写 |
| 系统 | Linux，kernel 5.15.0-186-generic |
| CPU / 内存 | 当前 shell 可见 12 CPU，约 31GiB 内存，约 25GiB available |
| GPU | RTX A6000，49,140 MiB，总量约 48GB |
| GPU 占用快照 | 1,360 MiB 显存已用，GPU utilization 0% |
| 驱动 | 570.211.01 |
| Home 所在磁盘 | `/data/disk1`：492G，99% 已用，约 6.3G 可用 |
| 根分区 | 194G，约 53G 可用；`/tmp`、`/var/tmp` 可写 |
| 默认 Python | `/usr/bin/python3`，3.10.12，未发现 NumPy/pandas/sklearn/torch/CatBoost/Transformers |
| 预装 Conda | `/opt/miniconda3`，base Python 3.14.6，同样未发现上述 ML 包，envs 为空；安装目录只读 |
| PyPI | server 上访问 NumPy simple index 返回 HTTP 200 |
| Hugging Face | server 上访问失败：Network is unreachable |
| 校内 AIGC API | server 上域名解析失败 |

课程说明中的 `/data/disk1/shared/aiaa3201` 目录存在且为共享材料。这是 server 接入资料所述目录，不代表已确认其中包含本项目两个任务；本项目数据以本机材料为准。

当前 GPU 可见，不等于训练栈已可运行。需要为本账号准备隔离的 Python 3.12 / CUDA 兼容环境，并先解决空间。根分区的临时目录可用于短期验证，但清理策略、空间配额和持久存放位置尚未确认，不能承担唯一成果副本。较大模型应在可访问的机器下载后校验并同步，不依赖 server 现场下载。

没有安装软件、启动训练、删除文件或停止其他用户进程。server 访问存在有效期，且可能临时停机，代码与结果以本机及 Git 仓库为主要保存位置。

本次 server 与 HPC 首次连接采用会话内 TOFU，未向用户全局 `known_hosts` 写入记录；尚无带外主机指纹核对。server 协商主机公钥的首见指纹为 `SHA256:2vd1ITyl6TVjRcCG5kBlXprj0jYesX3oAYKul5jIINo`。后续自动连接应先确认并固定可信主机公钥。

## HPC：首次认证检查，结论已更正

按[官方一二期登录文档](https://docs.hpc.hkust-gz.edu.cn/docs/hpc12/connection)和[三期登录文档](https://docs.hpc.hkust-gz.edu.cn/docs/hpc3/connection)，分别测试：

- `hpc1login.hpc.hkust-gz.edu.cn`
- `hpc2login.hpc.hkust-gz.edu.cn`
- `hpc3login.hpc.hkust-gz.edu.cn`

三处均完成 SSH 握手，但首次检查错误地将 group 的 IP 字段拼入密码，并把 group 账号用于学校入口，因此当时的 `Authentication failed` 不能判断任何一个正确环境是否可用。

更新连接资料并正确解析后，group SSH、学校门户与 SSH 均已成功。学校 Slurm 分区、GPU、账号 association/QOS 与提交权限已检查，具体结果及短作业记录见 [HPC 对比](hpc-comparison.md)。个人存储配额、正式训练环境及收费政策仍待确认。

官方三期文档还说明存在外网访问限制，并提供校内软件镜像与环境迁移方案；若未来采用三期，依赖与权重分发也需要单独准备。参考：[三期用户手册](https://docs.hpc.hkust-gz.edu.cn/docs/hpc3/)。

## 课程 API：鉴权与最小调用通过

按[官方接入文档](https://docs.hpc.hkust-gz.edu.cn/docs/aigc/api-usage/)核对，当前 V2 接口采用 OpenAI 兼容协议与 Bearer 鉴权：

- Base URL：`https://aigc-api.hkust-gz.edu.cn/api/v2`
- `GET /models`：本机使用现有课程 key 返回 HTTP 200。
- `POST /chat/completions`：`Qwen3.8-27B` 最小请求返回 HTTP 200，有 choices 和 usage 字段。
- 本次 smoke test：56 input tokens，16 output tokens，合计 72 tokens；`max_tokens=16`。仅验证服务调用，不涉及数据集、训练或正式评估。

平台当前列出 Qwen / DeepSeek 及其他厂商模型。课程限制只允许满足截止日期要求的 Qwen / DeepSeek；模型别名与平台价不能直接替代课程要求的正式模型版本和 2026-10-21 官方标准输入/输出单价。正式模型选型、历史价格证据、额度与速率限制仍待 SPEC 后核实。其他提供商的本地 key 只确认存在变量，未额外调用。

官方[平台模型列表及定价](https://docs.hpc.hkust-gz.edu.cn/docs/aigc/api-model-list)用于核对当前接口名称；不作为已经核实的课程历史官方计费表。

## 收到 SPEC 后的推荐推进方式

初步推荐：**本机作为开发与 Agent 控制环境，按需通过 SSH 使用 Linux GPU worker**。理由是本机现有工具链完整、课程 API 已通；server 有更大显存，但网络和存储不适合直接承担全部流程。HPC 暂列备选，认证恢复后重新评估。该方案尚未实现，最终依据 SPEC 的运行、部署和提交要求确定。

优先处理会改变整体实现的问题：

1. 将 SPEC 和课程要求对齐，明确统一任务输入、运行入口、自治范围、时间/计算预算、提交预测接口与日志格式。
2. 固定允许的模型版本与历史价格证据；所有 Agent、失败请求与重试纳入调用记录和预算控制，确认计费接口返回的 token 字段含义。
3. 明确生成代码的执行边界、超时、资源限制、失败恢复与结果回传。控制器与训练执行器的分离需要保持整个实验决策闭环自主。
4. 按数据类型与实验反馈选择方法，建立训练/验证/最终评估的清楚边界，统一记录候选方案、指标、代码版本、关键决策、随机种子和预测产物。
5. 先完成一个从任务输入到可运行预测产物的自主闭环，再扩展多模态任务和性能优化；在干净环境及最终冻结版本上验证复现。

这轮准备不预先锁定 Agent 框架，也不针对已发布任务编写专用解法。8GB 本机 GPU 足以开展接口开发与小规模验证；需要更大模型或批量训练时，再根据实测配置 A6000 worker。

## Git 与交付状态

已通过 GitHub 网页建立 private 仓库 `hhuang999/aiaa3102-ml-super-agent`。本机 GitHub SSH 身份鉴权成功；现有 `gh` CLI token 返回 HTTP 401，使用 SSH push 可以避开对该 token 的依赖。

首次版本管理包含作业要求、原始课程任务、项目入口、环境检查脚本和本记录。`.env`、连接凭据、带 key 的接入说明、临时探测文件及生成产物由 `.gitignore` 排除。后续提交前继续检查敏感信息与 diff。

新增环境检查脚本已在 `vb` 下通过 CUDA 与 train/val 结构检查，标准库诊断模式也在系统 Python 下通过。原始 CSV 含行尾空格，保留其原貌；`.gitattributes` 阻止数据文件自动转换换行。对本轮新增文档与脚本单独执行 diff whitespace 检查。
