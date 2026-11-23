# 报告代理设计与边界

`superagent_adapter.course.reporting.generate_report(evidence, output_dir, writer=None)` 创建一个**不存在**的新目录，生成中文 Markdown 草稿、结构化报告、证据、准备好的写作上下文和人工复核记录；调用写作器时另有 `llm_calls.jsonl`，拒绝时保留 `rejected_writer_output.json`。返回各主要产物路径及模式。调用者负责准备 JSON 兼容的汇总字典，必须包含 `schema_version`, `task`, `policy`, `core_status`, `core_summary`, `collection`, `evaluation`, `provenance`。应明确标记真实运行或夹具、验证策略是否待定、测试分数是未请求还是不可用。允许 `task.rows`、`evaluation.rows` 计数、`policy.test_label_boundary` 元数据和 `evaluation.predictions_sha256` 哈希。

流程为 Core 字段过滤 → 有界上下文 → 可选写作器 → 格式和 JSON pointer 引用校验 → 渲染。首先深拷贝输入；Core 摘要/状态中带有 code、rows、labels、api_key 等受限名称的字段递归移除，路径写入 `review.json.omitted_core_fields`。原始调用方输入不变，CLI 的 `adapter/evidence.json` 与 `delivery/core_report/` 继续保留原始 Core 记录；报告目录下的 `evidence.json` 是过滤后的报告证据。其他事实分区仍严格校验，不会为生成报告而跳过检查。字段过滤不能识别普通自由文本中的所有敏感内容，调用方仍需审核，不能声称已完全脱敏。

上下文每个必需分区约 5000 字符，Core 摘要在约定的各小节独立分配总计约 5000 字符预算。列表最多 8 项、对象最多 30 字段、字符串最多 500 字符。长时间线及关键事件均匀抽样、保留首尾，转换为以原始列表索引为键的对象，例如 `/core_summary/score_timeline/24/score` 仍引用原始第 25 项，不会重编号。抽样不保证覆盖所有重要事件；提示词明确要求不能把抽样项当成连续实验。记录内优先保留节点、步骤、分数、事件等简短字段，较长说明随后裁剪。`writer_context.json` 保存实际准备的 payload，`review.json` 记录截断与选择；即使未启用 LLM 也会保存这份准备好的上下文，不能据此宣称已发起调用。

写作器协议：`writer(payload)` 返回 `sections` 列表（每项包含 `title`, `text`, `evidence_refs`），以及 `usage`, `model`, `call_id`, `latency_secs`。引用必须存在于写作器实际收到的上下文。事实区和指标由程序渲染，写作器无法覆盖；有效引用不证明叙述真实，人工复核始终必需。调用或校验失败自动回退并记录原因。CLI 发现报告配置缺失、损坏或类型非法时生成 `deterministic_configuration_error` 报告，明确未发起模型调用，不生成虚假的调用账目。没有写作器时模式为 `deterministic_no_llm`。请勿将模型叙述用于自动决策。

可选 `superagent_adapter.course.report_llm.make_chat_writer(base_url=..., model=..., api_key_env=...)` 是单次、无重试的 OpenAI 兼容 chat completions 适配器；需要显式指定网关、模型和存有密钥的环境变量名称。禁止自动重定向，超时必须大于零且不超过 120 秒。不会默认使用任何外部服务；仅应在获得授权后调用。请求可能产生费用，费用预算和执行控制由团队网关负责，本模块不实现这些控制。调用记录使用独立生成的 call_id，包含平面 token 用量与调用、输出校验状态；未知用量记为 null，不是零。网络失败不会记录服务端原始错误；畸形模型响应如有 usage 会保留计数。提供的输入和输出文件应由调用者保存在适当的访问控制下，包括保留的被拒绝写作器输出。

这是独立的 Markdown/JSON 草稿设计，最终课程 PDF 的整合及真实数据验证尚待完成；不构成正式合规结论。

## 0.2.0 的一致性与来源记录

生成报告时重新比较 Core 状态和摘要 best 中的明确节点/分数字段，结果保存到 `review.json.summary_consistency`。`checked_fields`、`conflicts[].path` 是相对于原始 Core 摘要的路径。没有可信锚点或已知字段时记为 `not_verifiable`，不能声称已经一致。发生冲突时，报告证据中的 core_summary 为空，Writer 获知冲突且不得猜测方法；原始摘要仍保留在调用方的 adapter/evidence.json 与 Core 归档中。验证分数继续来自已收集的状态，最终分数来自 Adapter 评价结果。

报告新增“本次运行来源”区，展示启动前记录的版本和源码标识。缺失信息写未提供；静态发现与来源快照不构成真实模型或完整环境的复现验收。CLI 会把生成结果写入整体 outcome，并记录报告对应的输入证据哈希；评分后旧报告标为 stale，生成失败重试不会丢掉上一份报告的路径。

## Core 1.1 报告兼容

状态 `best_score` 和摘要 `best.score` 表示最佳单模型的验证分数；最终选择通过 collection 的 `selected_result` 单独展示，包含来源、类型、方法、指标、验证分数、节点与原因。最终选择可能是集成，验证分数可能缺失；测试 accuracy 始终独立展示。报告同时显示完整脚本、组合重放及仅推理是否经过验证，不会由文件存在推断验证成功。

一致性检查对存在的可信状态锚点比较 `best` 和 `final` 对应字段，最终分数比较 `final.validation.score`。未提供的字段不计入检查；状态明确为 null 时，摘要不得声称该字段有值。若提供最终成员，将与已收集的最终方案节点进行核对；`consistent` 仅覆盖列出的已检查字段；冲突仍排除整份摘要。报告证据和写作输入均移除 `core_summary.task.data_preview`，因为该字段可能包含原始样本。Core 1.1 的 final、control、diagnostics 分区拥有独立的有限上下文预算；原始记录仍由调用方管理。
