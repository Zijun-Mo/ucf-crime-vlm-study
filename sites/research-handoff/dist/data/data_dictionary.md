# 结果数据字典

数据快照覆盖20个正式条件，其中18个原有条件于2026-11-15冻结，两项E9消融于2026-11-20完成。JSON／CSV指标保持[0,1]；网页显示百分比。差值网页显示百分点(pp)。不存在的单视频指标为null，不将全正常或全异常视频的定位指标记作0。

- summary.json / summary.csv：统一条件ID、中文标签、模型、规模、方法、提示词、窗口时长、10项指标及区间。主表展示其中6项。
- test_videos.json：固定290视频的name、category、label、fps、n_frames、intervals及源视频SHA。intervals是官方1-based闭区间；[-1,-1]无效，截断到实际末帧。
- predictions/{condition}.json：每视频的start_s、raw_logits、来源身份与哈希。拼图原文件storyboard字段导出为raw_logits，数值不变；*.js是相同数据的浏览器离线加载包装。
- reports/{condition}.json：原口径评分报告，包括每项value、ci95、固定有效视频集合和逐视频结果。
- per_video/{condition}.json / per_video.csv：逐视频帧数、异常帧数、窗口max／mean／top-k分数、within_auroc／within_ap。
- comparisons.json、statistics/：固定差值方向和配对区间。bootstrap=1000，seed=42，按视频成簇配对；未统一校正多重比较。
- training/：原生教师1419份分数、来源与元数据、训练／验证／预留校准划分、9轮开发预测与指标、近似时间标注、目标和抽样规则、选模／停止证据。E2／E5／E9同一条轨迹。
- analysis/：两类负帧分解、分层统计、逐视频变化、固定min／median／max案例规则和图片来源。
- provenance/：原报告及原始预测文件SHA、配置和最终验收摘要。导出文件的SHA见files.sha256；原始文件SHA与重新序列化导出文件SHA不同，二者各自明确记录。
- history/：早期精度／缓存／训练协议结果，不与正式主表混合；历史只作追溯，不代表当前运行状态。
- assets/figures/：PNG／SVG真实结果图；assets/cases/为少量带时间戳模型输入联系表。

## 重建与指标

窗口覆盖区间为[round(start_s*fps), min(n_frames, round((start_s+duration)*fps)))；每帧取覆盖窗口原始logit算术均值。正式步长1秒，所有帧必须有覆盖。2／4／8秒窗口分别使用对应duration，不能统一按4秒重建。

视频标签由Normal类别为0、其余为1；主视频聚合为窗口max。Top-k敏感性聚合取全部测试窗口中最高ceil(N/8)分数的均值；这不同于训练时抽样bag的top-k。

within_macro_auroc / within_macro_ap：对140个混合标签视频分别算AUROC／AP，再等权平均。pair-weighted Within按视频异常帧数×正常帧数加权；anomaly-only pooled AUC仍含跨视频对，它们都不等于Macro。

AUROC并列给0.5；AP使用sklearn average_precision_score的阶梯积分口径。bootstrap每次抽290个视频，所有方法共用同一抽样权重；Macro只对抽中有效混合视频重新归一化。

所有曲线的min–max仅为显示；图形分桶保留极值，不参与正式计算。教师软标签由sigmoid(完整教师logits)逐异常视频min–max得到；它是训练目标，不能与显示归一化或测试逐帧重建混淆。

消融逐窗口分数使用同一预测格式。`data/analysis/e9_ablation.json` 保存同一290视频上的点估计、与 E9 的配对区间及按负帧来源分解，`e9_ablation_protocol.json` 记录消融定义和匹配抽样的哈希。
