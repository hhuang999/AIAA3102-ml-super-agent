> Historical reference for the standalone Core 0.1 adapter. For the active Core 1.1 integration, use COURSE_ADAPTER.md.

# Adapter 决策与交接（2026-11-12）

依据用户最新确认、《Adapter接口for superagent.md》及课程项目要求。聊天中新的职责分工优先于接口文档仍写“Adapter 暂定负责费用”的旧条目。

| 决策 | 当前实现 | 交接时需说明 |
| --- | --- | --- |
| 验证策略 | core_default；保留原 train/val，均可见，Adapter 不重新划分 | **暂定默认，之后可能修改，必须团队商议**；Core 记录真实验证过程 |
| 测试标签访问 | 公共任务与 private 路径分离，任务不含原始压缩包和测试标签 | **暂定服务器路径适配**，同 UID 非 OS 隔离；部署再议 |
| 最终选择 | 运行前固定 best；配置可改 ensemble | 不用最终测试分数选方案；ensemble 不可用时保留明确失败 |
| 可运行方案 | 完整 solution.py，模型产物可缺省 | 不承诺只加载模型预测；真实完整脚本仍需实际复现 |
| Core 异常退出 | 验证仍存在的方案与预测，标 partial | 有效部分结果和正常完成区分；无方案也生成失败报告 |
| 报告职责 | 我们设计输入证据、Writer、校验和渲染 | Agent 写叙述，程序负责指标和事实区，团队负责最终审阅/PDF |
| 费用与运行资源 | 外部团队模块处理 | Report Agent 自身调用必须记账并汇总，不能漏算或记未知为零 |
| 接口样例 | 自编 Core fixture，版本 0.1 | 同学真实样例尚待核验；不更改 Core 架构 |
| 支持范围 | 两个发布任务；CSV/NPY；accuracy | 不将不同指标或图片任务默认为 accuracy；新格式需补适配 |
| 正式评测资格 | 当前所有回执 false | 小规模模拟分数不是课程成绩，尚未进行正式冻结/预算核验 |

## 首次真实样例需核对

1. `python -m superagent.run` 的参数及输出根目录是否一致；指定运行环境能否导入模块。
2. 真实 `status.json`、`run_summary.json` 的嵌套字段和类型。报告只接收摘要，原始 journal/代码归档，不直接送给写作器。
3. `top_solution/topN/` 包含 solution.py、metric.txt、node_id.txt、submission.csv；最终预测与 sample_submission 的目标列/ID 一致。
4. 目前 `members.json` 按本地样例接受列表，或 `{"members": [...]}`；成员使用 `node_id`、`validation_score`、`weight`。真实键名若不同，应加显式转换并补测试，不能静默猜测。
5. 在约定的 input/、working/、submission/ 工作目录运行真实 solution.py，确认生成预测与保存方案对应。此操作可能重新训练，执行前需商议计算安排。
6. 最终接入团队模型/记账模块，再完成真实 Report Agent 写作和完整自主运行。报告调用同样受课程模型和费用限制。

## 当前配置不包含人工算法选择

`configs/headlines.json` 与 `configs/coupon.json` 仅描述文件格式、字段、标签语义和评分指标；同一 `prepare_task`、Core 调用、收集与报告逻辑处理全部任务。不能据任务名选择预写模型或超参。

## 交付前检查对象

- 不是只看返回码：检查 status、节点/验证分数绑定、最终 ID、类别值、脚本、ensemble 所有成员。
- 不是只看文件存在：预测/产物封存 SHA-256，评分前重新核对；不把这些哈希当作强制安全边界。
- Report Agent 不从测试标签选择方案。最终指标由程序计算，缺失区分 `not_requested` 与 `unavailable`。
- 文本引用检查只能保证引用存在，不能自动证明写作语义正确。原始摘要自由文本还需审阅，不能宣称所有潜在敏感内容都能被字段过滤捕捉。
- 本地模拟验证的是约定和失败路径；真实 Core、真实模型网关与课程正式结果仍分开验收。
