# UCF-Crime VLM 异常检测研究

AIAA 3111 Group 11：视觉语言模型的视频分类能力，能否迁移成帧级异常检测？

**[中文研究说明与结果网页](https://ucf-crime-research-atlas.intactopera4.chatgpt.site)** · [公开快照范围](PUBLICATION.md) · [数据与复算说明](sites/research-handoff/dist/README.md) · [提案公开副本](docs/proposal.md)

本仓库包含实验实现、测试、冻结预测与分析数据、网页源码。四个研究维度为决策适配、弱监督、提示词（包括零样本拼图视觉上下文）和模型规模。结果冻结于2026-11-15，共18个正式条件，每条件290视频、37,188窗口、1,111,808帧。教师和开发数据供分析与复核使用，不含原视频或模型权重。

## 主要发现

- **决策适配：**Open-Jev通用决策适配在当前受控实验中未呈现一致明确收益。
- **弱监督：**旧MIL改善视频识别和汇总帧级AUROC，却损害视频内定位。冻结min–max软标签改善整体检测，并缓解相对旧MIL的定位退化；相对零样本严格定位不退化尚未确立。
- **提示词：**P1/P2/P3没有统一最优策略；全局拼图没有在完整测试集上重现开发阶段的明显收益，帧级AP下降。
- **模型规模：**9B未显示稳定优势，应结合显存、吞吐及成本解释。

| Qwen条件 | 汇总帧级AUROC | 汇总帧级AP | 视频内Macro AUROC | 视频级Max AP |
|---|---:|---:|---:|---:|
| 零样本P1 | 80.35 | 37.12 | 75.77 | 85.49 |
| 旧MIL | 84.61 | 33.87 | 70.79 | 95.62 |
| 零样本＋拼图 | 80.61 | 31.70 | 75.81 | 87.65 |
| 软标签E9 | 86.69 | 41.30 | 76.67 | 94.14 |

单位为百分比。完整18条件、六项主要指标、按视频配对的1000次bootstrap区间及边界见网页和[汇总数据](sites/research-handoff/dist/data/summary.json)。E2/E5/E9来自同一训练轨迹，不是三个独立重复。

## 无GPU复核全部主表

```sh
git clone https://github.com/Zijun-Mo/ucf-crime-vlm-study.git
cd ucf-crime-vlm-study
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r sites/research-handoff/dist/requirements.txt
cd sites/research-handoff/dist
python verify_results.py --verify-files
python verify_results.py --all --bootstrap 0
python verify_results.py --condition qwen_localization_v4_plateau_test --bootstrap 1000 --paired qwen_zs_p1_4s
```

预期：文件校验通过；18条件均覆盖290视频、1,111,808帧，复算与冻结点估计一致；最后一条复核E9与零样本的配对区间。CPU复算使用完整原始窗口logit及覆盖关系；无需原视频、模型、账号或云端权限。它不等于重新执行模型推理或重训。

## 阅读和打包网页

直接打开`sites/research-handoff/dist/index.html`可离线阅读。也可在该目录运行`python -m http.server 8775`后访问本机端口。下载压缩包是生成文件，未重复提交Git；生成方法：

```sh
python sites/research-handoff/package_results.py
```

所有未压缩结果已包含在仓库中。在线站点提供完整结果包及离线网页包。

## 实验代码与运行范围

| 位置 | 内容 |
|---|---|
| `src/openjev_dss/` | 评分、MIL、教师约束、窗口独立软标签、抽样修正、恢复和评价实现 |
| `tests/` | CPU数值、窗口独立性、评分与恢复等回归测试 |
| `research/*.py` | 历史及正式推理/训练workers、性能探测和分析入口 |
| `research/*/policy.json` | 原实验协议；资源配置是历史记录，不代表当前应启动的作业 |
| `scripts/` | 分析、绘图、审计及原Colab执行编排 |
| `experiments/` | 更早期实验workers；与正式18条件分开解释 |
| `sites/research-handoff/dist/data/` | 正式窗口分数、统计、训练开发结果、协议与来源哈希 |

若需开发测试，在独立环境安装`python -m pip install -e '.[dev,analysis]'`，再运行`python -m pytest -q`。发布时187项测试通过。GPU运行还需要与原实验匹配的Transformers/PEFT等环境、公开UCF-Crime视频、公开模型、原生/规范化视频与视觉缓存、自己的云盘和Colab配置。私有的收据、缓存、权重和云端目录均不在仓库内；已有worker中的历史路径与身份校验不应被视为即开即跑的通用部署配置。详见[运行边界](docs/REPRODUCING.md)。

原始结果导出脚本需要完整原实验档案；无需运行它即可读取、打包或复核已提交数据。公开数据保留原始档案文件哈希作为来源，不声称所有来源档案都随仓库分发。

## 数据和披露

视频内指标为140个混合标签视频逐视频等权Macro，不等于帧对加权Within-AUROC。开发时间标注是助手审核的近似区间，不是官方真值，仅用于验证和分析。案例画面是公开基准的少量分析证据，不据此推断事件起因。AI使用与审核记录在`data/provenance/`；方法参考在`data/references.bib`。

公开快照移除了个人标识、账户文件和私人云盘链接，未携带原始Git历史。第三方模型、数据与参考代码的使用仍受其各自条款约束；本仓库不重新授权这些材料。
