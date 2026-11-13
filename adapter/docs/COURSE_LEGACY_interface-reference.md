> Historical reference for the standalone Core 0.1 adapter. For the active Core 1.1 integration, use COURSE_ADAPTER.md.

# SUPERAGENT_INTERFACE：适配智能体与核心智能体的接口约定

## 1. 这份文档的作用

项目分成两部分，由两个人分别开发：

| 名称                 | 负责人 | 做什么                                    |
| ------------------ | --- | -------------------------------------- |
| **适配智能体**（Adapter） | A   | 接收课程任务，整理成标准任务目录，调用核心智能体，再把结果放到课程要求的位置 |
| **核心智能体**（Core）    | B   | 读取标准任务目录，自动搜索并运行多个方案，输出最好的方案和全部记录      |

这份文档只规定两者之间的边界：Adapter 交给 Core 什么，Core 交还给 Adapter 什么。Core 内部怎么搜索不在这里说明。

只要双方都遵守这份文档，两部分可以同时开发，互不等待。

## 2. 一次调用的全过程

1. Adapter 准备一个任务目录（第 4 节）。
2. Adapter 运行一条命令启动 Core（第 3 节）。
3. Core 运行，直到做满步数或用完时间。运行中它不需要任何人工输入。
4. Core 把结果写到输出目录（第 6 节），然后退出。
5. Adapter 读取输出目录，完成课程要求的后续工作。

一次调用只处理一个任务。要处理多个任务，Adapter 多次调用 Core，每次用不同的输出目录。

整体流程：

```
课程给的数据和说明
   → Adapter 整理成标准任务目录
   → Adapter 的 run.py 启动 Core
   → Core 搜索、运行、改进方案
   → Core 输出方案、模型文件、预测、分数、日志、报告摘要
   → Adapter 算最终分数，写报告
```

课程要求的统一入口 `run.py` 属于 Adapter。Core 是被它调用的一个模块。

## 3. 调用方式

### 3.1 命令行

```bash
python -m superagent.run \
  --task-dir   /path/to/task_dir \
  --output-dir /path/to/output_dir \
  [--config /path/to/config.yaml] \
  [--time-limit-secs 43200] \
  [--max-steps 500] \
  [--higher-is-better true|false] \
  [--num-gpus 1] [--num-cpus 8] [--parallel 2] \
  [--keep-artifacts-top-k 5] \
  [--seed 42]
```

默认值按当前的运行环境设定，见第 10 节。

|参数|必填|含义|默认值|
|---|---|---|---|
|`--task-dir`|是|标准任务目录，见第 4 节|无|
|`--output-dir`|是|结果目录。必须不存在或为空|无|
|`--config`|否|配置文件，可覆盖 Core 的内部参数|Core 自带的默认配置|
|`--time-limit-secs`|否|总时间上限（秒）。到时间后 Core 自己停止并写完结果|43200（12 小时）|
|`--max-steps`|否|最多生成多少个方案节点|500|
|`--higher-is-better`|否|评价指标是否越大越好。不提供时，Core 让 LLM 从任务说明里判断|不提供|
|`--num-gpus`|否|可用的 GPU 数量|1|
|`--num-cpus`|否|可用的 CPU 核数。Core 把它们平分给并行运行的方案|8|
|`--parallel`|否|同时运行几个方案。它们共用 GPU|2|
|`--keep-artifacts-top-k`|否|为前几名方案保留模型文件，见 6.3|5|
|`--seed`|否|随机种子|42|

### 3.2 Python 函数

Adapter 和 Core 在同一个 Python 环境里，所以也可以直接调用函数。参数名和命令行一一对应。

```python
from superagent import run_task

status = run_task(
    task_dir="tasks/leaf",
    output_dir="runs/leaf_001",
    time_limit_secs=43200,
    higher_is_better=True,
)
# status 的内容和 output_dir/status.json 相同
```

这个函数会一直阻塞到 Core 结束。它不抛出异常来表示搜索失败，结果一律看 `status["status"]`。

建议优先用命令行方式。这样 Core 在独立进程里运行，它出错或被终止都不会影响 Adapter。

### 3.3 环境变量

|变量|含义|
|---|---|
|`SUPERAGENT_LLM_API_KEY`|LLM 接口的密钥|
|`SUPERAGENT_LLM_BASE_URL`|LLM 接口的地址|
|`SUPERAGENT_PRETRAIN_DIR`|本地预训练模型目录。计算节点不能联网时，生成的脚本从这里加载模型|

模型名称（用哪个 Qwen 或 DeepSeek 模型）写在配置文件里，不在命令行。

**建议：** 强烈建议 Adapter 提供 `--higher-is-better`。如果让 LLM 判断并且判断错了，整次运行的方案排序都会反过来。

## 4. 输入：标准任务目录

```
task_dir/
├── description.md            必须
└── data/                     必须
    ├── sample_submission.csv 强烈建议
    └── ...                   训练数据和不带标签的测试数据，结构不限
```

### 4.1 `description.md`

这是 LLM 了解任务的唯一文字来源。它必须包含下面四项内容，格式不限。

|内容|要写清楚什么|例子|
|---|---|---|
|任务目标|要预测什么，是二分类还是多分类，有几个类别|“根据叶片照片，预测 5 种状态中的一种”|
|数据说明|`data/` 里每个文件或文件夹是什么；哪一列是 ID，哪一列是标签|“`train.csv` 有两列：`image_id` 和 `label`。图片在 `train_images/`”|
|评价指标|指标名称，越大越好还是越小越好，必要时给出公式|“准确率，越高越好”|
|提交格式|提交文件的列名，每列是标签还是概率|“两列：`image_id`、`label`。`label` 是 0 到 4 的整数”|

不要写的内容：时间限制、网络限制、排名和奖励。Core 会自动删掉这类内容，写了也没有用。

如果 Adapter 用 LLM 来生成这个文件，有两点要求：

- 这些 LLM 调用也计入课程的费用限额。Adapter 应该用 6.4 里 `llm_calls.jsonl` 的同一种格式记录自己的调用，方便最后合并统计。
- LLM 写的说明可能把指标方向写错或写得含糊。所以 Adapter 应该另外用 `--higher-is-better` 明确传入方向。

### 4.2 `data/`

- 文件类型和目录结构不限。可以有 CSV、JSON、图片文件夹、文本文件。
- 可以放 `.zip` 压缩包。Core 会自动解压。
- Core 把这个目录当作只读。它不修改这里的任何文件。

### 4.3 `data/sample_submission.csv`

Core 用它检查每个方案的预测文件是否合格。要求：

- 第一列是测试样本的 ID，包含**全部**测试样本，每个样本一行。
- 其余各列是预测列。列名和顺序就是最终提交文件要求的列名和顺序。
- 预测列的值可以随便填，例如全部填 0。
- 文件名必须是 `sample_submission.csv`。

Core 的检查规则是：方案的预测文件必须和它有相同的列名、相同的列顺序、相同的行数、相同的 ID 集合，并且预测列没有空值。

不提供这个文件时，Core 跳过格式检查。这样有风险：一个格式错误的预测文件可能被选成最好方案。

### 4.4 数据隔离（Adapter 的责任，非常重要）

课程规定：指定的测试集只能用于最终评估，不能用于训练、调参或选择方案。

Core 生成的脚本可以读取 `data/` 里的**所有**文件。所以 Adapter 必须保证：

1. **`data/` 里不能有测试集的标签。** 测试数据只保留特征和 ID。
2. **课程给的评估集，不管叫 test 还是 validation，只要它用于最终评分，它的标签就不能放进 `data/`。** 原因：Core 发现名字里带 `val` 的文件并且里面有标签时，会建议 LLM 把它并入训练集。
3. 方案内部需要的验证集，由 Core 生成的脚本自己从训练数据里划分。Adapter 不需要准备。

例子：课程给了 `train.csv`、`test.csv`（带标签，用于最终评分）。Adapter 应该把 `train.csv` 原样放进 `data/`，把 `test.csv` 去掉标签列后放进 `data/`，标签自己另外保存，等 Core 结束后再用来算最终分数。

## 5. Core 运行时会做什么（供 Adapter 了解）

- Core 在输出目录下建一个工作区，把 `data/` 链接进去。
- 每个方案是一个独立的 Python 脚本。它读取数据，训练模型，在自己划分的验证集上算分，预测全部测试样本，写出预测文件。
- Core 根据验证分数比较方案。它从不接触测试标签。
- Core 按规则反复改进方案，直到做满步数或用完时间。

## 6. 输出：结果目录

```
output_dir/
├── status.json                  运行摘要，Adapter 应该先读它
├── report/
│   ├── run_summary.json         给 Adapter 写报告用的摘要，见 6.6
│   └── run_summary.md           同样内容的可读版本
├── best_solution/
│   ├── solution.py              最好方案的完整脚本
│   ├── metric.txt               它的验证分数
│   ├── node_id.txt              它的节点编号
│   └── artifacts/               它训练出的模型文件，见 6.3
├── best_submission/
│   └── submission.csv           最好方案对测试集的预测
├── top_solution/
│   └── top1/ … top20/           前 20 名候选。每个目录有 solution.py、submission.csv、
│                                metric.txt、node_id.txt；前 5 名另有 artifacts/
├── ensemble/                    前几名的加权合并，见 6.5
│   ├── submission.csv
│   ├── members.json
│   └── ensemble.py
├── logs/
│   ├── journal.json             全部方案节点的记录
│   ├── decisions.jsonl          每一步的决策记录
│   ├── llm_calls.jsonl          每一次 LLM 调用的记录
│   └── superagent.log           运行日志
├── global_memory/
│   └── records.json             经验记录
└── workspace/                   Core 的内部工作区，结构不保证稳定
```

### 6.1 `status.json`

Core 每完成一个节点就更新这个文件。所以即使 Core 被强行终止，这个文件也是最新的。

```json
{
  "interface_version": "0.1",
  "status": "success",
  "stop_reason": "time_limit",
  "started_at": "2026-11-20T09:00:00+08:00",
  "updated_at": "2026-11-20T21:00:03+08:00",
  "elapsed_secs": 43203,
  "higher_is_better": true,
  "metric_direction_source": "cli",
  "num_nodes": 187,
  "num_valid_nodes": 142,
  "num_branches": 6,
  "best_node_id": "a3f9…",
  "best_metric": 0.9012,
  "best_has_artifacts": true,
  "ensemble_available": true
}
```

|字段|取值|
|---|---|
|`status`|`running`：还在运行。`success`：至少有一个合格方案。`no_valid_solution`：正常结束但没有合格方案。`failed`：Core 自身出错|
|`stop_reason`|`max_steps`、`time_limit`、`interrupted`、`error`|
|`metric_direction_source`|`cli`：来自命令行参数。`llm`：由 LLM 判断|
|`best_metric`|验证分数，不是测试分数。没有合格方案时为 `null`|

### 6.2 `best_solution/` 和 `best_submission/`

- `solution.py` 是一个完整脚本，包含预处理、训练、验证和测试集预测。
- `submission.csv` 就是这个脚本在 Core 运行期间生成的预测文件。两者是一一对应的。
- 每当出现新的最好方案，Core 就更新这两个目录。所以运行中途终止，这里仍然是到那一刻为止的最好结果。

重新运行 `solution.py` 的方法：把工作目录切到一个包含 `input/`（即 `data/` 的内容）、`working/`、`submission/` 三个子目录的文件夹，然后运行脚本。脚本把预测写到 `./submission/submission.csv`，并在最后打印一行 `Final Validation Score: <分数>`。

### 6.3 模型文件 `artifacts/`

Core 要求每个方案脚本把预测所需的全部东西保存到一个固定文件夹。这包括训练好的模型权重，以及预处理用到的对象，例如编码器、分词器、标签映射。

- Core 为验证分数排前 5 名的方案保留这个文件夹，并复制到对应的 `artifacts/` 目录。其余方案的模型文件会被删除，以节省磁盘。数量由 `--keep-artifacts-top-k` 控制。
- 文件格式不固定，由方案自己决定。例子：神经网络是 `.pth` 或 `.safetensors`，梯度提升树是 `.txt` 或 `.joblib`。
- `artifacts/` 里的文件就是生成 `submission.csv` 的那一次运行留下的，三者一一对应。
- 个别方案可能没有按要求保存模型。这时 `artifacts/` 是空的，方案仍然算合格。`status.json` 的 `best_has_artifacts` 说明最好方案有没有模型文件。
- Core 不提供单独的“只预测”脚本。预测所需的预处理和推理代码都在 `solution.py` 里。

### 6.4 `logs/`

**`journal.json`**：一个列表，每个元素是一个方案节点。

|字段|含义|
|---|---|
|`id`、`parent_id`、`branch_id`|节点编号、父节点编号、所属分支|
|`stage`|节点类型：`draft`、`improve`、`debug`、`evolution`、`fusion`、`fusion_draft`|
|`plan`|LLM 写的计划|
|`code`|完整脚本|
|`term_out`|运行时的终端输出（过长时截断）|
|`metric`|验证分数；出错时为 `null`|
|`is_buggy`|是否被判为出错|
|`analysis`|对运行结果的文字总结|
|`exec_time`|运行耗时（秒）|
|`created_at`|创建时间|

**`decisions.jsonl`**：每行一步。

|字段|含义|
|---|---|
|`step`、`ts`|第几步，时间|
|`selected_node_id`|选中了哪个节点来扩展|
|`selection_mode`|`uct`（探索）或 `topk`（利用）|
|`action`|选了哪种操作：`draft`、`improve`、`debug`、`evolution`、`fusion`、`aggregation`|
|`reason`|为什么选这个操作，例如 `branch_stagnant`|
|`new_node_id`|生成的新节点|

**`llm_calls.jsonl`**：每行一次 LLM 调用。

|字段|含义|
|---|---|
|`ts`、`call_id`|时间，调用编号|
|`agent`|哪个智能体发起的调用，例如 `planner`、`coder`、`result_parser`|
|`node_id`|这次调用为哪个节点服务；与节点无关时为 `null`|
|`model`|模型名称|
|`prompt_tokens`、`completion_tokens`|接口返回的输入和输出 token 数|
|`latency_secs`|耗时|
|`ok`|调用是否成功。失败和重试的调用也各记一行|

Core 只记录 token 数，不计算费用，也不做预算控制。Adapter 可以用这个文件按官方价格算出费用。

### 6.5 `ensemble/`

Core 在结束时把前几名方案的预测文件合并成一个。数值列（例如概率）按验证分数加权平均，类别列（例如标签）加权投票。这个功能默认开启。

|文件|内容|
|---|---|
|`submission.csv`|合并后的预测|
|`members.json`|参与合并的方案：节点编号、验证分数、权重、使用的合并方法|
|`ensemble.py`|一个小脚本。它读取各成员的预测文件，按 `members.json` 重新算出合并结果|

注意四点：

- 合并后的预测**没有验证分数**。它只是把测试集预测合在一起，Core 没有办法在验证集上给它打分。它是否比单个最好方案更好，只能由 Adapter 用测试标签算出来。
- 课程规定测试集不能用于选择方案。所以 Adapter 必须在运行前就固定最终报告哪一个结果（单个最好方案，或合并结果），不能算完两个测试分数后挑高的那个。另一个可以作为对比写进报告。
- 合并结果对应的“可运行方案”是：各成员的 `solution.py` 加上 `ensemble.py`。
- 合格方案少于 2 个时不做合并，`status.json` 的 `ensemble_available` 为 `false`。

### 6.6 报告摘要 `report/run_summary.json`

这个文件专门给 Adapter 写报告用。它由代码从 `journal.json`、`decisions.jsonl`、`llm_calls.jsonl` 汇总得到，不调用 LLM，所以没有额外费用，内容也可以核对。Adapter 写报告时读这一个文件就够了，不需要自己解析完整日志。

|字段|内容|报告里可以怎么用|
|---|---|---|
|`task`|指标方向及其来源，检测到的数据类型，数据预览文字|任务和数据分析|
|`run`|起止时间，停止原因，节点总数、合格数、出错数，分支数|运行概况|
|`best`|最好方案的节点编号、验证分数、计划、方法摘要、从初稿到它的完整改进路径|最终方案说明|
|`score_timeline`|每次刷新全局最好成绩的记录：第几步、时间、节点、分数、用了哪种操作、改了什么|分数随时间变化的曲线|
|`branches`|每个分支：起始思路、节点数、最好分数、最好节点|比较不同路线|
|`actions`|每种操作（起草、改进、调试、进化、融合、聚合）：次数、成功率、平均分数变化|说明哪些机制起了作用|
|`key_events`|重要事件列表：分支停滞并触发进化或融合、多分支聚合、发现数据泄漏、调试成功、探索转向利用|“根据反馈自主改进”的证据|
|`top_solutions`|前 20 名：节点编号、分数、分支、操作类型、方法摘要、是否有模型文件|候选方案表|
|`ensemble`|成员、权重、合并方法|合并结果说明|
|`failures`|出错节点按原因分类的数量，每类一个例子|局限性和失败分析|
|`resources`|LLM 调用次数和输入、输出 token 数，按智能体和按模型分别统计；脚本总运行时间|资源使用，费用计算|

`run_summary.md` 是同样内容的可读版本，方便人直接看。

和 `status.json` 一样，这两个文件在每个节点完成后更新。Core 被中途终止时，它们反映到那一刻为止的情况。

## 7. 退出码和异常情况

|退出码|含义|输出目录的状态|
|---|---|---|
|0|正常结束，至少有一个合格方案|完整|
|2|正常结束，没有合格方案|没有 `best_solution/` 和 `best_submission/`，日志完整|
|1|Core 自身出错|`status.json` 的 `status` 为 `failed`，日志保留到出错为止|
|3|输入不合格，例如缺少 `description.md`|只有 `status.json` 和错误说明|

- **超时：** 到达 `--time-limit-secs` 后，Core 停止启动新方案，终止还在运行的脚本，写完结果后退出。
- **强行终止：** Adapter 可以随时终止 Core 进程。`status.json`、`best_solution/`、`best_submission/` 和日志在每个节点完成后都会更新，所以终止后它们仍然可用。
- **部分结果：** 课程对未完成的任务也按有效的部分结果给分。Adapter 应该在退出码不为 0 时仍然读取输出目录。

## 8. 责任划分

| 事项                         | Adapter | Core     |
| -------------------------- | ------- | -------- |
| 理解课程任务的原始格式                | ✔       |          |
| 写出 `description.md`        | ✔       |          |
| 保证 `data/` 里没有测试标签         | ✔       |          |
| 提供 `sample_submission.csv` | ✔       |          |
| 设计、运行、比较、改进方案              |         | ✔        |
| 选出最好方案                     |         | ✔（按验证分数） |
| 记录方案、决策、Core 的 LLM 调用      |         | ✔        |
| 保存前几名方案的模型文件               |         | ✔        |
| 汇总报告摘要 `run_summary.json`  |         | ✔        |
| 记录 Adapter 自己的 LLM 调用      | ✔       |          |
| 用指定测试集算最终分数                | ✔       |          |
| 计算费用、控制预算                  | ✔（暂定）   |          |
| 课程要求的 `run.py`、文件位置        | ✔       |          |
| 写报告                        | ✔       |          |

Core 不使用任务名或数据集名来选择方法。它只根据 `description.md` 的内容、数据本身的特征和运行结果做决定。Adapter 也不应该通过 `description.md` 向 Core 传递针对某个具体任务预先准备的解法。


## 9. 运行环境和功能一览

**当前运行环境**

|项目|值|
|---|---|
|GPU|1 块|
|CPU|8 核|
|每个任务的总时间|12 小时|
|并行运行的方案数|2 个，每个方案分到 4 个 CPU 核，两个方案共用 GPU|
|Python 环境|Adapter 和 Core 共用一个环境，依赖合并到一份 `requirements.txt`|

**Core 提供的功能**

|功能|说明|章节|
|---|---|---|
|自动搜索方案|读取任务目录，自动生成、运行、比较、改进多个方案，全程不需要人工输入|5|
|预测格式检查|对照 `sample_submission.csv` 检查每个方案的预测文件|4.3|
|单个最好方案|脚本、预测文件、验证分数、模型文件|6.2、6.3|
|前 20 名候选|每个有脚本、预测文件、验证分数；前 5 名有模型文件|6、6.3|
|加权合并结果|前几名预测的加权平均或投票，附可重现的合并脚本|6.5|
|完整记录|全部方案节点、每一步决策、每一次 LLM 调用|6.4|
|报告摘要|由代码汇总的一份文件，供 Adapter 写报告|6.6|
|中途终止安全|每个节点完成后更新全部结果文件，随时终止都能拿到部分结果|7|

**Core 不提供的功能**

见第 8 节的责任划分和第 9 节。最主要的三项是：读取课程原始格式、用测试标签算最终分数、写报告。

## 10. 最小示例

输入：

```
tasks/leaf/
├── description.md
└── data/
    ├── train.csv
    ├── train_images/
    ├── test_images/
    └── sample_submission.csv
```

调用：

```bash
python -m superagent.run \
  --task-dir tasks/leaf \
  --output-dir runs/leaf_001 \
  --time-limit-secs 21600 \
  --higher-is-better true
```

Adapter 之后的处理：

```python
import json, pathlib, shutil

out = pathlib.Path("runs/leaf_001")
status = json.loads((out / "status.json").read_text())

if status["status"] == "success":
    shutil.copy(out / "best_submission/submission.csv", "results/leaf/predictions.csv")
    shutil.copytree(out / "best_solution", "results/leaf/best_solution")   # 脚本和模型文件
    shutil.copytree(out / "top_solution", "results/leaf/top_solution")
    if status["ensemble_available"]:
        shutil.copytree(out / "ensemble", "results/leaf/ensemble")

summary = json.loads((out / "report/run_summary.json").read_text())
# 然后用自己保存的测试标签计算最终分数，再根据 summary 写报告
```