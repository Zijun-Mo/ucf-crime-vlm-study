# AIAA 3111 — Group 11 Project Proposal

## 1. Project Title（项目名称）

**中文题目：** 基于 UCF-Crime 的视觉语言模型视频异常检测研究：决策适配、弱监督、提示词与模型规模

**English Title:** Vision-Language Models for Video Anomaly Detection on UCF-Crime: Decision Adaptation, Weak Supervision, Prompting, and Model Scale

**Subtitle:** A Comparative Study of Between-Video Recognition and Within-Video Localization

## 2. Group

AIAA 3111 Group 11. Personal identifiers are omitted from the public copy.

## 3. Background and Problem Statement（研究背景与问题定义）

视频异常检测（Video Anomaly Detection, VAD）旨在从监控视频中识别偏离正常行为或场景规律的事件，并尽可能确定异常发生的时间位置。与普通分类任务相比，VAD 存在异常样本稀缺、逐帧标注昂贵、场景变化较大等困难。传统方法往往需要针对特定数据或监督设定进行训练；预训练视觉语言模型（Vision-Language Models, VLMs）则提供了利用通用视觉语义知识进行零样本异常判断的可能性。

近年来的多模态决策模型使另一种研究路线变得可行：不依赖长文本生成，而是为每个视频窗口直接输出可排序的异常分数。本项目以原始 Qwen3.5-2B 与其通用决策适配模型 Open-Jev-2B 为已有研究基础，探讨通用决策能力能否迁移到 VAD，以及何种 VAD 专项监督与输入策略能够改善异常判断。

然而，VAD 的评价存在一个容易忽略的差异：模型可能善于判断“哪个视频包含异常”，却未必能准确指出“异常出现在视频的什么时刻”。将所有测试视频的帧合并计算 Pooled Frame-level AUROC，可能无法充分揭示这两种能力。因此，本文不以单一 AUROC 排名作为结论，而将视频间异常识别（Between-video Recognition）和视频内异常时间定位（Within-video Localization）作为两类独立研究对象。

**总体研究问题：**

在公平且可复现的对照条件下，通用决策适配、VAD 弱监督学习、自然语言提示词策略和模型规模分别如何影响视频间异常识别与视频内异常定位？这些因素对两类能力的影响是否一致？

据此，项目设置四个具体子问题：

- **RQ1（Decision Adaptation）：** 与原始视觉语言模型相比，通用决策适配方案是否改善视频间识别与视频内定位？
- **RQ2（Weak Supervision）：** 利用视频级标签进行 LoRA+MIL 弱监督训练后，两类能力分别发生怎样的变化？其收益是否取决于基础决策方案？
- **RQ3（Prompting）：** 不同语义侧重点的自然语言 Prompt 是否对视频间识别和视频内定位产生不同影响？
- **RQ4（Scale）：** 在统一评价协议下，Qwen3.5-2B、9B 及可选的 27B 模型在异常识别、时间定位和计算成本方面呈现怎样的关系？

研究目标是解释各类因素的实际作用和局限，而不是预设任何方法一定更优，亦不以刷新 SOTA 为必要条件。

## 4. Proposed Methodology / Technical Approach（研究方法与技术路线）

### 4.1 Dataset and Preprocessing（数据集与预处理）

本项目以公开的 UCF-Crime 数据集为主要实验基础，沿用官方训练/测试划分。既有工作已完成官方测试集 290 段视频、1,111,808 帧的全量处理与评估。实验使用固定时长的视频窗口进行推理，已有主要设置为 4 秒窗口、2 FPS 帧采样率及 224×224 输入分辨率。前期已完成 1、2、4 秒窗口步长的敏感性分析。模型为每个窗口输出异常分数，再按窗口覆盖关系映射回逐帧序列。

后续研究将冻结视频清单、时间标注边界、预处理规则与评分脚本，对各模型提供一致输入。所有训练和 Prompt 选择只利用与测试集隔离的训练/开发数据。考虑到官方 UCF-Crime 测试集已用于先前探索，本文将透明披露其开发历史；如需更严格的独立泛化验证，将采用未参与配置选择的外部测试数据。

### 4.2 Model and Experimental Factors（模型与四类因素）

**因素一——通用决策适配（Decision Adaptation）**

使用原始 Qwen3.5-2B 作为预训练多模态基线，并使用恢复原生视频能力的 Open-Jev-2B 作为通用决策适配方案。Open-Jev 在 Qwen 多模态底座上结合已发布的决策 LoRA 与 Scalar Decision Head，为每个视频窗口输出异常分数；原始 Qwen 则使用 Yes/No 候选读出。

该直接对照评估的是完整决策方案的变化，不能单独归因于 LoRA。本项目将重点比较两种方案在视频间异常识别和视频内时间定位上的差异。

**因素二——VAD 弱监督（Weak Supervision）**

采用 Multiple Instance Learning（MIL）弱监督学习方法，利用视频级正常/异常标签进行训练，无需逐帧异常标注。模型以视频窗口为实例，基于 MIL 损失学习异常评分。

为控制训练成本，主要采用参数高效微调（Parameter-Efficient Fine-Tuning, PEFT），通过 LoRA 更新部分模型参数，而不进行全量微调。

核心实验采用“原始 Qwen／Open-Jev × 无专项训练／LoRA+MIL”的 2×2 对照设计：

| 模型方案          | Zero-shot | LoRA+MIL |
| ------------- | --------- | -------- |
| 原始 Qwen3.5-2B | 已完成       | 待完成      |
| Open-Jev-2B   | 已完成       | 已完成      |

现有 Open-Jev+LoRA+MIL 训练结果作为基础，后续补齐 Qwen+LoRA+MIL，并尽可能统一训练数据、LoRA 配置、训练预算与评价协议。

主要分析弱监督训练对两类检测能力的影响，以及不同决策方案下弱监督收益是否存在差异。对于 Yes/No Token-logit 与 Scalar Decision Head 的结构差异，将在结果解释中明确说明。

已有 MIL+DSS-inspired loss 实验作为辅助消融保留，不作为核心研究因素。

**因素三——自然语言提示词策略（Natural-language Prompt Strategies）**

参考 ASK-HINT 等视频异常检测研究，本项目在不更新模型参数的条件下设计三类自然语言 Prompt，比较不同语义引导对异常识别与时间定位的影响：

- **P1 Generic（简洁异常判断）：** 沿用现有零样本实验中的原始简洁异常判断 Prompt，作为历史基线。
- **P2 Context and Normality-guided（场景规范与异常定义）：** 引导模型结合可见场景的正常行为与安全规范判断异常，检验场景上下文的作用。
- **P3 Fine-grained Action and Temporal-cue（细粒度动作与时序线索）：** 引导模型关注具体动作、人—物交互以及短期时序变化，检验细粒度线索是否有助于异常检测和时间定位。

实验初期固定 2B 模型规模，在原始 Qwen 与 Open-Jev 两种决策方案上开展配对比较。三组 Prompt 保持一致的输出格式与评分规则，正式测试前固定具体措辞，不使用测试集标签进行提示词选择。

本研究关注自然语言 Prompt Engineering，而非更新可学习参数的 Soft Prompt Tuning。

**因素四——模型规模（Model Scale）**

模型规模实验采用 Qwen3.5 同系列多模态模型，明确比较三个参数档位：

- **Qwen3.5-2B：** 小规模基线，复用现有零样本评估工作。
- **Qwen3.5-9B：** 必须完成的第二规模，用于与 2B 直接比较。
- **Qwen3.5-27B：** 可选的第三规模，在计算资源与项目时间允许时开展。

主实验采用统一的 Zero-shot 设置，固定 Prompt、视频预处理与评价协议，不要求额外训练对应规模的 Open-Jev 适配器。

除异常识别与时间定位表现外，同时记录推理耗时、吞吐量和 GPU 显存峰值，分析模型规模与检测效果之间的 Accuracy–Cost Trade-off。

考虑到不同规模可能伴随训练或架构差异，研究结论将限定为同系列模型的受控比较。

### 4.3 Main Experiment Matrix（主要实验矩阵）

| 模块      | 实验设置                                                       | 主要回答的问题           |
| ------- | ---------------------------------------------------------- | ----------------- |
| A：F1×F2 | Qwen-ZS、Open-Jev-ZS、Qwen-LoRA+MIL、Open-Jev-LoRA+MIL（均为 2B） | 通用决策适配、弱监督及其交互    |
| B：F3    | Qwen3.5-2B 与 Open-Jev-2B × P1/P2/P3                        | 提示词语义策略对两类能力的影响   |
| C：F4    | Qwen3.5-2B vs. 9B；可选 27B                                   | 模型规模与检测效果、计算成本的关系 |

模块 A 为必须完成的核心对照；模块 B 为提示词敏感性实验；模块 C 至少完成 2B 与 9B 的比较，27B 为可选增强项。

项目不要求四因素全面排列组合，以控制工作量并保证每组比较具有明确的解释对象。

### 4.4 Evaluation and Statistical Analysis（双层评估与统计）

**视频间识别指标簇：**

以 Video-level AUROC 为主指标，以 Video-level AP 和不同异常类别下的视频级表现为辅助指标。

视频级异常分数沿用预先固定的窗口 Max 聚合方式，并以 Mean/Top-k 聚合作敏感性分析，避免事后逐模型择优。

**视频内定位指标簇：**

以 Within-video Macro AUROC 为主指标，以 Within-video Macro AP、代表性视频异常分数时间曲线为辅助评价；在具有适当时间标签与独立阈值选择条件时，补充事件级指标。

Within-video 评价仅针对同时存在正常与异常时段的视频，并在所有模型间使用同一固定视频集合。

**标准 Benchmark 指标：**

另外单列 Pooled Frame-level AUROC 和 Pooled AP，保持与 VAD 常见 Benchmark 报告方式的联系，但不将 Pooled AUROC 直接等同于某一个指标簇。

**补充敏感性分析：**

本项目将视频窗口步长（Window Stride）与时间窗口长度（Window Duration）共同作为补充敏感性分析，研究时间处理配置对异常检测性能及计算成本的影响：

- **窗口步长敏感性分析（已完成）：** 在固定 4 秒窗口长度下，已完成 1 秒、2 秒及 4 秒窗口步长的对照评测，用于研究窗口重叠程度与时间评分密度对异常检测性能的影响。
- **时间窗口长度敏感性分析（计划开展）：** 以现有 4 秒窗口为基线，在计算资源允许时比较 2 秒、4 秒和 8 秒窗口，固定帧采样率及窗口步长等条件，分析时间上下文范围对视频间异常识别和视频内异常定位的影响。

两项实验均作为核心四因素之外的补充分析，不单独扩展主实验矩阵。

**统计分析与公平性：**

结果将同时报告指标点估计和以视频为单位的配对 Bootstrap 95% 置信区间，对因素的两类效果差异进行分析，并结合计算成本、异常类别及成功/失败案例解释结果。

为保证公平，实验将保留模型版本、精度类型、窗口和帧采样、Prompt、训练/验证划分、随机种子、LoRA 参数配置、可训练参数数量、GPU 时间及显存等配置。

对未充分统一条件的历史成绩，将明确标为阶段性结果，而不把其差值直接解释为单个因素的因果效应。

### 4.5 Preliminary Results（已完成的前期工作）

项目已恢复 Open-Jev-2B 的原生视频推理能力，并完成 UCF-Crime 官方测试集全部 **290 段视频、1,111,808 帧**的零样本评测。以 1 秒窗口步长为例，原始 Qwen3.5-2B 与 Open-Jev-2B 的 Pooled Frame-level AUROC 分别为 **80.39% 和 80.85%**，AP 分别为 **37.07% 和 34.40%**，表明通用决策适配尚未展现出一致的性能优势。

已完成 1、2、4 秒窗口步长的敏感性分析。以 4 秒步长为例，原始 Qwen 与 Open-Jev 的 AUROC 分别降至 **78.87% 和 79.31%**，AP 分别降至 **34.07% 和 31.01%**。两种模型均表现出步长增大、检测指标下降的趋势，且 Open-Jev 的 AUROC 略高但 AP 较低；差异的统计显著性仍待检验。

在已有的弱监督初轮训练中，Open-Jev+LoRA+MIL 取得 **83.683% AUROC、35.374% AP**；加入 DSS-inspired loss 后取得 **83.477% AUROC、34.809% AP**。由于部分实验设置尚未完全统一，这些结果仍需进一步公平对照。

探索性时间定位分析发现，原始 Qwen 与 Open-Jev 在 140 段可计算视频上的 Within-video Macro AUROC 分别约为 **75.47% 和 75.15%**，进一步支持分别研究视频间识别与视频内定位的必要性。

## 5. Expected Outcomes and Timeline（预期成果与时间安排）

### 5.1 Expected Outcomes（预期成果）

1. **公平、可复现的四因素研究结果：** 以核心 2×2 对照为基础，研究通用决策适配与 LoRA+MIL 弱监督的作用，再纳入 Prompt 与 Scale 模块。

2. **双层视频异常检测评价协议：** 明确区分视频间识别与视频内时间定位，同时保留 Pooled Frame-level AUROC 作为标准 Benchmark 参考。

3. **可解释的比较图表：** 包括因素对两类主指标的影响图、统计置信区间、Prompt/Scale 变化曲线、窗口步长与时间窗口长度敏感性分析、计算效率对比，以及典型时间定位成功与失败案例。

4. **研究局限与影响因素分析：** 包括完整决策方案与 LoRA 作用的区别、弱监督标签粒度、Prompt 选择偏差、模型规模与架构混杂，以及已有测试集使用历史。

5. **课程交付成果：** 完成 Poster、6–10 页正文的研究报告、训练/推理/评价代码、README、数据获取和复现说明及 AIGC Disclosures。

### 5.2 Timeline（时间安排）

| 时间         | 计划工作与提交节点                                       |
| ---------- | ----------------------------------------------- |
| 10 月 15 日前 | 完成小组组建及 Proposal 定稿、提交；归档已有零样本、MIL 及窗口步长敏感性分析结果 |
| 10 月下旬     | 冻结数据与评测协议，补齐 Qwen+LoRA+MIL 及双层指标分析              |
| 11 月上旬     | 开展 P1/P2/P3 Prompt 对照及时间窗口长度敏感性分析               |
| 11 月中旬     | 完成 Qwen3.5-2B 与 9B 规模对照；资源允许时尝试 27B，完善统计图表      |
| 11 月 27 日  | 提交 Poster.pdf                                   |
| 12 月 4 日   | Poster Session 现场展示与答辩                          |
| 12 月 11 日前 | 提交 Report.pdf、代码、README、数据与复现说明                 |

## References

[1] Sultani, W., Chen, C., & Shah, M. (2018). Real-World Anomaly Detection in Surveillance Videos. *CVPR*, 6479–6488.

[2] ZefanCai. (2026). Open-Jev-2B. *Hugging Face*. [https://huggingface.co/ZefanCai/Open-Jev-2B](https://huggingface.co/ZefanCai/Open-Jev-2B)

[3] Zou, S., et al. (2026). Unlocking Vision-Language Models for Video Anomaly Detection via Fine-Grained Prompting. *WACV*, 4223–4233.
