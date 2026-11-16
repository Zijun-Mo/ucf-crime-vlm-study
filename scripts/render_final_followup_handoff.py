"""Render the epoch5 formal evidence without overwriting epoch2 predictions."""
import datetime,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from openjev_dss.followup_budget import spending,amount


def read(path):return json.loads((ROOT/path).read_text())


def main():
    dest=ROOT/'results/future_work_v1';final=read('results/qwen_localization_v4_epoch5_test/summary.json')
    early=read('results/qwen_localization_v4_epoch2_test/summary.json')
    comparison=read('results/qwen_localization_v4_epoch5_test/method_comparisons.json')
    transfer=read('results/future_work_v1/frame_transfer_analysis.json')
    per_video=read('results/future_work_v1/per_video_transfer.json');arm='minmax_soft_epoch5'
    native=read('results/study_formal_v2/qwen_zs_p1_4s.json');mil=read('results/study_formal_v2/qwen_formal_mil_p1_4s.json')
    teacher=read('results/qwen_localization_v2_epoch1_test/summary.json');montage=read('results/storyboard_context_formal_v1/summary.json')
    selected=read('research/localization_v4_continuation/selected_checkpoint.json');stop=read('research/localization_v4_continuation/operator_user_stop.json')
    policy=read('research/localization_v4_epoch5_test/policy.json');budget=spending(ROOT)
    point=lambda r:{k:v['value'] for k,v in r['metrics'].items()}
    keys=['pooled_auroc','pooled_ap','within_macro_auroc','video_max_ap']
    labels=['汇总帧级 AUROC','汇总帧级 AP','视频内 macro AUROC','视频级 Max AP']
    rows=[('零样本',point(native)),('旧抽样 top-2 MIL',point(mil)),('教师约束 V2，第1轮',teacher['metrics']['candidate']),('零样本＋拼图',montage['metrics']['storyboard']),('冻结 min-max 软标签，第2轮',early['metrics']['candidate']),('冻结 min-max 软标签，第5轮（最终选择）',final['metrics']['candidate'])]
    lines=['# Qwen 后续实验：最终结果、分析与复现证据','',f'更新：{datetime.datetime.now(datetime.timezone.utc).isoformat()}。工作范围为实验、分析和结论材料，海报由报告团队处理。','',
        '提案13个必做条件与4个后续正式预测条件均已完成。最终第5轮模型在完整测试集上，相对零样本提高汇总帧级AUROC、帧级AP和视频级AP；三项名义95%配对区间高于零。视频内AUROC点估计提高0.80个百分点，区间仍跨零，不能证明定位严格不退化。','',
        '第5轮与第2轮来自同一条训练轨迹。第5轮的帧级AUROC、视频内AUROC和视频级AP点估计较高，帧级AP略低；这四项直接比较区间均跨零，不能据此确认延长训练有稳定的额外收益。','',
        '## 1. 统一正式结果','',
        '每个条件均覆盖相同290个视频、37188个窗口和1111808帧。Qwen使用P1、4秒窗口、1秒步长。帧分数为覆盖该帧的原始窗口logit均值；视频级AP固定使用窗口最大值。视频内macro指标对140个同时含正常与异常帧的视频等权平均。表内单位为百分比。','',
        '|条件|汇总帧级 AUROC|汇总帧级 AP|视频内 macro AUROC|视频级 Max AP|','|---|---:|---:|---:|---:|']
    for name,value in rows:lines.append('|'+name+'|'+'|'.join(f'{100*value[k]:.2f}' for k in keys)+'|')
    lines += ['', '原13条件的提示词、窗口长度、Open-Jev和规模对照见[主矩阵](../study_formal_v2/README.md)与[原验收](../study_formal_v2/FINAL_ACCEPTANCE.md)。不再扩展模型规模、种子或消融。','', '![检测与定位点估计](frame_transfer.png)','',
        '### 第5轮减零样本','', '|指标|差值（百分点）|视频配对95%区间（百分点）|','|---|---:|---|']
    def add_intervals(values):
        for key,label in zip(keys,labels):
            d=values[key];ci=d['ci95'];lines.append(f'|{label}|{100*d["difference"]:+.2f}|[{100*ci["low"]:+.2f}, {100*ci["high"]:+.2f}]|')
    add_intervals(final['paired_contrasts'])
    lines += ['', '![第5轮相对零样本的区间](../qwen_localization_v4_epoch5_test/paired_differences.png)','',
        '按视频成簇、配对1000次bootstrap，种子42。它描述固定模型在视频重采样下的不确定性，不包含重训随机性、重新标注或多种子差异。没有预注册非劣效界限，也未执行非劣效检验；区间跨零不能证明等效或不退化。','',
        '### 第5轮减第2轮：延长训练的证据','', '|指标|差值（百分点）|视频配对95%区间（百分点）|','|---|---:|---|']
    add_intervals(comparison['contrasts']['minmax_soft_minus_minmax_soft_epoch2'])
    lines += ['', '主指标点估计提高约1.25个百分点，但直接区间下界约−0.012个百分点，仍包含零，不能因四舍五入或点估计较高而称为已确立收益。两轮检查点不是独立复现。','',
        '### 第5轮减旧MIL：分类与定位的取舍','', '|指标|差值（百分点）|视频配对95%区间（百分点）|','|---|---:|---|']
    add_intervals(comparison['contrasts']['minmax_soft_minus_old_mil'])
    lines += ['', '视频内AUROC比旧MIL高约5.78个百分点，区间高于零；视频级AP仍低约1.52个百分点，区间低于零，不能称分类等效。汇总帧级AUROC/AP点估计分别高约0.56/6.95个百分点，但直接区间均跨零，不能宣称这两项显著优于旧MIL。方法间比较为冻结模型后的描述性比较，未做多重比较校正。','',
        '## 2. 分类收益转移到了哪些帧','',
        '测试集有84338个异常帧、648905个正常视频帧、378565个异常视频背景帧。把所有异常帧分别与后两组负帧比较，按负帧数量加权，可精确重建汇总AUROC；权重为63.16%与36.84%。第二组仍包含不同异常视频间的配对，不能等同于视频内macro AUROC。','',
        '|条件|异常帧 对 正常视频帧 AUROC|异常帧 对 异常视频背景帧 AUROC|','|---|---:|---:|']
    for name in ['zero_shot','old_mil','teacher_constraint','minmax_soft','minmax_soft_epoch5']:
        row=transfer['results'][name];lines.append(f'|{name}|{100*row["anomaly_vs_normal_video_auroc"]:.2f}|{100*row["anomaly_vs_abnormal_background_auroc"]:.2f}|')
    contribution=read('results/future_work_v1/auc_contributions.json')['contributions'][arm]
    lines += ['', f'第5轮的汇总AUROC提升中，正常视频比较贡献约{contribution["normal_video_comparison_pp"]:+.2f}个百分点，异常视频背景比较贡献约{contribution["abnormal_background_comparison_pp"]:+.2f}个百分点，合计约+4.82。收益主要来自正常视频与异常帧的区分。该精确分解不是某个损失项的因果证据。','',
        '同一视频内部正负帧对只占全部正负配对的约0.3879%，所以汇总AUROC对跨视频分数差异很敏感。按帧对数加权的视频内指标与按视频等权的macro指标也不可混用。','',
        '![汇总AUROC贡献](frame_auc_decomposition.png)','',
        '在各模型约80%异常帧召回的描述性工作点，误报率如下。阈值只用来解释已冻结预测，不用于选模型或部署校准。','',
        '|条件|正常视频误报率|异常视频背景误报率|','|---|---:|---:|']
    for name in ['zero_shot','old_mil','minmax_soft','minmax_soft_epoch5']:
        groups=transfer['results'][name]['score_groups'];lines.append(f'|{name}|{100*groups["normal_video"]["false_positive_rate_at_80pct_anomaly_recall"]:.2f}%|{100*groups["abnormal_background"]["false_positive_rate_at_80pct_anomaly_recall"]:.2f}%|')
    lines += ['', '第5轮在相同召回下明显减少正常视频误报，相对旧MIL也减少异常视频背景误报；相对零样本，异常视频背景误报并未同步减少。固定logit零阈值时召回和误报会一起改变，不能只看均值或某一个工作点宣称全面改善。完整分位数、均值和零阈值比例见[分解数据](frame_transfer_analysis.json)。','',
        '## 3. 退化特征与不确定性','']
    values=[r['within_deltas'][arm] for r in per_video];up=sum(v>1e-12 for v in values);down=sum(v< -1e-12 for v in values);tie=len(values)-up-down
    lines += [f'140个混合异常测试视频中，第5轮相对零样本：{up}个视频内AUROC提高、{down}个下降、{tie}个基本不变（绝对差≤1e-12）。平均点估计上升不意味着每个视频都保持。','',
        '|分层|视频数|旧MIL减零样本（百分点）|第5轮减零样本（百分点）|第5轮探索性95%区间（百分点）|','|---|---:|---:|---:|---|']
    for field in ['fraction_bin','duration_bin']:
        for name,row in transfer['stratified_within_differences'][field].items():
            ci=row['exploratory_ci95'][arm];interval='样本不足' if ci is None else f'[{100*ci["low"]:+.2f}, {100*ci["high"]:+.2f}]'
            label=('异常占比 ' if field=='fraction_bin' else '时长 ')+name
            lines.append(f'|{label}|{row["videos"]}|{100*row["mean_within_deltas"]["old_mil"]:+.2f}|{100*row["mean_within_deltas"][arm]:+.2f}|{interval}|')
    lines += ['', '上述第5轮异常占比与时长分组区间均跨零，不能确认某一组稳定改善或退化。Explosion类别（n=21）的平均变化约−2.02个百分点、区间跨零；Stealing（n=5）约+7.43个百分点，样本较少。全部类别的样本量与区间保留在JSON中；未校正多重比较，不作因果解释。','',
        '## 4. 固定案例与画面','',
        '按第5轮减零样本的视频内AUROC差值，固定选取最小、中位、最大三例，文件名处理并列；仅用于解释结果，不用于调整模型。','',
        '![第5轮固定案例](../qwen_localization_v4_epoch5_test/case_curves_shape.png)','',
        '曲线逐视频归一化只用于显示排序形状；所有正式指标使用原始logit。[原始曲线](../qwen_localization_v4_epoch5_test/case_curves_raw.png)同时保留。红带为原基准异常时间区间。','',
        '|案例规则|视频|时长秒|异常占比|零样本视频内AUROC|第5轮视频内AUROC|变化百分点|','|---|---|---:|---:|---:|---:|---:|']
    cases=read('results/qwen_localization_v4_epoch5_test/case_selection.json')['cases']
    base={r['name']:r for r in final['reports']['baseline']['per_video']};candidate={r['name']:r for r in final['reports']['candidate']['per_video']}
    for role,label in [('worst','最大下降'),('median','中位变化'),('best','最大改善')]:
        c=cases[role];name=c['name'];lines.append(f'|{label}|{name}|{c["duration_s"]:.2f}|{100*c["anomaly_fraction"]:.2f}%|{100*base[name]["within_auroc"]:.2f}|{100*candidate[name]["within_auroc"]:.2f}|{100*c["within_delta"]:+.2f}|')
    lines += ['',
        '- [Explosion027画面](../qwen_localization_v4_epoch5_test/case_frames/Explosion027_x264.mp4.jpg)：室内座椅附近出现短暂白光/烟雾，随后有人起身、移动。曲线在标注结束后的背景也有较高分，最终模型第二个背景峰更强；不能据此认定具体错误机制。',
        '- [Shooting022画面](../qwen_localization_v4_epoch5_test/case_frames/Shooting022_x264.mp4.jpg)：等候区域中，约95秒附近有人抬臂、多人迅速躲避，之后人员离开。低清图片不足以确认具体器具；曲线整体排序接近零样本，对应中位变化约+0.29个百分点。',
        '- [Arson009画面](../qwen_localization_v4_epoch5_test/case_frames/Arson009_x264.mp4.jpg)：车辆附近短暂强光/起火，后续仍可见燃烧；画面内时间戳有跳变，基准只把较短一段标为异常，后续燃烧画面落在基准背景。解读时须保留这种标注边界，不能自行把它重标成异常，也不能据单帧宣称模型理解了因果。','',
        '画面来自远端固定224×224、2fps规范化模型输入，保留源视频SHA、规范化SHA和真实采样时间；原片与权重留在远端。[取图规则和SHA清单](../qwen_localization_v4_epoch5_test/case_frames/manifest.json)可复核。Codex已逐图检查3张联系表，不把低帧率联系表称为完整原生视频审阅。旧第2轮案例、曲线和画面继续保留。','',
        '## 5. 训练、停止与选模型','',
        'V4从原生Qwen3.5-2B初始化rank8/alpha16、dropout0的LoRA；P1、4秒窗口、1秒步长、BF16、AdamW学习率1e-5，每步8视频。教师是同一固定原生零样本模型。异常视频全部教师sigmoid分数先逐视频min-max成软标签，正常目标为0，使用窗口独立BCE；不加入focal、Huber、时间平滑或额外定位网络。极小分数跨度的异常视频按固定规则跳过该视频正监督。仅借鉴[MIST的局部软伪标签思路](https://openaccess.thecvf.com/content/CVPR2021/html/Feng_MIST_Multiple_Instance_Self-Training_Framework_for_Video_Anomaly_Detection_CVPR_2021_paper.html)，不声称复现完整MIST。','',
        '目标和归一化系数在抽样前确定；每视频每轮均匀抽25%且最多64窗口，固定包含概率修正估计完整视频均值。固定目标下，单窗口分数梯度为固定权重乘p−y，其他窗口抽样不会改变该项损失或分数梯度；共享参数、批梯度相加与统一裁剪仍影响整体更新。1256训练视频每轮157次更新，验证覆盖163视频；112视频帧级开发集包含34个助手审核近似时间标注的异常视频和78个正常视频，时间标注不进入训练损失。','',
        '第2轮因原15CU额度停止并完成一次正式测试；之后用户追加最多20CU要求续训到开发平台期，从328步完整AdamW/RNG断点继续。第3至5轮开发结果如下，均不是测试成绩。','',
        '|轮次|开发帧级AUROC|开发帧级AP|开发视频内AUROC|验证视频Max AP|','|---|---:|---:|---:|---:|']
    for epoch in [2,3,4,5]:
        report=read(f'results/{"localization_v4_dev" if epoch==2 else "localization_v4_continuation_dev"}/epoch_{epoch}_comparison.json')['trained']
        lines.append('|'+str(epoch)+'|'+'|'.join(f'{100*report[k]:.2f}' for k in ['dev_pooled_auroc','dev_pooled_ap','within_macro_auroc','video_max_ap'])+'|')
    lines += ['',
        '用户在看到第4、第5轮视频内AUROC连续下降后，明确要求停止，并确认仍按最高合格帧级AUROC选择第5轮做完整测试。这是观察开发结果后新增的停止依据；原连续两轮无至少0.2个百分点合格帧级AUROC改善的平台期规则尚未满足，因此不能称为自然收敛或预注册停止。三个基线守门条件保持不变。','',
        f'实际提交{stop["committed_steps"]}次更新，完成5轮验证；第5轮对应785次更新，第6轮21次未验证更新只保留供恢复并排除选模与测试。第5轮在测试前冻结，文件SHA为 `{selected["checkpoint_file_sha256"]}`，张量SHA为 `{selected["checkpoint_tensor_sha256"]}`。训练器第3/4/5轮约1.80/1.76/1.75小时，包含发布与冷准备的独立保守墙钟上界均低于3小时，见各轮核验JSON。','',
        '开发集连续下降是此次停止信号；完整测试第5轮的平均视频内AUROC点估计反而高于第2轮和零样本，相关差值区间均跨零。所以开发停止信号不能被写成已证明测试集发生负迁移。没有测试第3/4/6轮，不根据测试结果再选模型。','',
        '历史V3硬候选/focal路线的质量检查观测为73.90%/81.92%，未达到80%/90%的工程门槛，故未训练；这些阈值并非论文通用标准。用户随后明确授权了V4软标签路线。原失败和恢复记录保留：教师缓存源不完整时在零更新处核对并补齐；续训冷启动的kernel握手、视觉缓存恢复、原第2轮测试结果读取超时，以及取图时过期OAuth刷新均有日志。它们不被伪装成算法变化或成功收敛。','',
        '## 6. 支持与不支持的结论','',
        '- 支持：冻结第5轮模型相对零样本提高汇总帧级AUROC/AP及视频级AP，名义配对区间支持三项收益。',
        '- 支持：相对旧MIL，第5轮视频内排序更好，视频分类仍有小幅差距；全局帧级检测点估计更好，直接区间尚不能确认其优于旧MIL。',
        '- 支持：全局收益主要来自正常视频与异常帧的分离，异常视频背景仍是薄弱环节。',
        '- 尚不能支持：定位严格不退化、所有视频均提高、第5轮稳定优于第2轮、分类与旧MIL等效、某一个损失项/抽样方式的因果作用、多种子稳定性或SOTA。',
        '- 实验已按用户指令停止；不为追求正结果追加训练路线、检查点测试、种子或消融。','',
        '## 7. 复现、成本与AI使用','',
        '以下命令使用本地缓存，不占GPU；第一条还可完全离线复算正式主指标和配对区间。','', '```bash',
        'PYTHONPATH=src .venv/bin/python scripts/finalize_localization_test.py research/localization_v4_epoch5_test/20261115T093601Z/receipt.json --offline',
        'PYTHONPATH=src .venv/bin/python scripts/compare_frozen_localization_methods.py --epoch 5',
        'PYTHONPATH=src .venv/bin/python scripts/analyze_frame_transfer.py --include-teacher --include-soft --include-final',
        'PYTHONPATH=src .venv/bin/python scripts/render_final_localization_cases.py',
        'PYTHONPATH=src .venv/bin/python scripts/refresh_experiment_inventory.py',
        'PYTHONPATH=src .venv/bin/python scripts/render_followup_handoff.py','```','',
        '精确模型/输入/缓存/窗口配置见[冻结policy](../../research/localization_v4_epoch5_test/policy.json)和[注册](../../research/localization_v4_epoch5_test/registration.json)；源码归档、分配报价与GPU清理见[收据](../../research/localization_v4_epoch5_test/20261115T093601Z/receipt.json)。完整预测、SHA、逐视频数据、对比区间和案例在[第5轮结果目录](../qwen_localization_v4_epoch5_test/README.md)。训练恢复断点与全部权重仍在远端。','',
        '离线复算已实际执行，全部指标、配对bootstrap与逐视频报告与首次评分完全相同，见[复算审计](../../research/localization_v4_epoch5_test/offline_recomputation_audit.json)。最终[验收](../../research/future_work_v1/final_acceptance.json)已覆盖17个完成条件；现场[资源核验](../../research/localization_v4_epoch5_test/release_audit.json)确认无活动会话、分配0、计费速率0。训练与测试事件已收尾，监控退役，无待运行的已授权GPU工作。','']
    test_cost=amount(read('research/localization_v4_epoch5_test/20261115T093601Z/receipt.json'))
    lines += [f'本次第5轮测试估算 **{test_cost:.4f}/3 CU**。定位/拼图阶段合计约 **{budget["total_estimate_cu"]:.4f}/80 CU**，未用约 **{budget["remaining_cu"]:.4f} CU**。训练共约27.3220CU，其中续训约12.6216/20CU；两次V4正式测试合计约{budget["phases"]["v3_test"]:.4f}CU，案例CPU准备（含失败恢复）约{budget["phases"]["annotation_cpu"]:.4f}CU。采用报价乘观测时长估计，不是账号余额或精确逐笔账单，也不涵盖更早全部研究开销。','',
        'AI用于编码、调试、统计、分析草稿与画面审核。Hydrogen生成12个新增开发标注草稿，Codex结合视频呈现、拼图和边界证据逐条审核；34个异常开发区间为助手审核的近似标注，不是官方逐帧真值或独立人工双标。第5轮案例只作解释，未修改测试标注。实际使用与审核见[AI记录](../../research/localization_v4/ai_use_and_review.json)及[第5轮补充记录](../../research/localization_v4_epoch5_test/ai_use_and_review.json)。报告作者仍须独立核对数字、引用、画面和课程披露。','',
        '课程正文要求6至10页，最终ZIP截止12月11日23:59，时区以Canvas为准。交付范围为实验、分析和结论证据，不含海报制作；原提案保持原样，要求与源文件SHA见[课程契约](../../research/future_work_v1/course_contract.json)。']
    (dest/'README.md').write_text('\n'.join(lines)+'\n')
    summary_lines=['# 第5轮冻结模型完整测试','', '|指标|零样本|第5轮|','|---|---:|---:|']
    for key,label in zip(keys,labels):summary_lines.append(f'|{label}|{100*final["metrics"]["baseline"][key]:.2f}%|{100*final["metrics"]["candidate"][key]:.2f}%|')
    summary_lines += ['', '已核验290视频、37188窗口、1111808帧，固定第5轮，优化器更新为0。帧级AUROC/AP和视频级AP相对零样本的95%视频配对区间高于零；视频内AUROC区间跨零。','',
        '完整统计见[summary.json](summary.json)，与第2轮/旧MIL/教师约束模型的直接配对区间见[method_comparisons.json](method_comparisons.json)。','',
        '完整解释、失败分层、固定案例、训练停止依据、成本和CPU复算命令见[统一报告](../future_work_v1/README.md)。']
    (ROOT/'results/qwen_localization_v4_epoch5_test/README.md').write_text('\n'.join(summary_lines)+'\n')
    print('Final epoch5 handoff rendered')


if __name__=='__main__':main()
