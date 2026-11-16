"""Render final epoch9 evidence from frozen predictions and audited development."""
import datetime,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from openjev_dss.followup_budget import spending,amount


def read(path):return json.loads((ROOT/path).read_text())


def main():
    dest=ROOT/'results/future_work_v1';result_dir='results/qwen_localization_v4_plateau_test'
    final=read(result_dir+'/summary.json');direct=read(result_dir+'/method_comparisons.json')
    transfer=read('results/future_work_v1/frame_transfer_analysis.json');pv=read('results/future_work_v1/per_video_transfer.json')
    selected=read('research/localization_v4_plateau/selected_checkpoint.json');train=read('research/localization_v4_plateau/complete_training_acceptance.json')
    receipt_path='research/localization_v4_plateau_test/20261115T205551Z/receipt.json';receipt=read(receipt_path)
    budget=spending(ROOT);arm='minmax_soft_epoch9'
    keys=['pooled_auroc','pooled_ap','within_macro_auroc','video_max_ap'];labels=['汇总帧级 AUROC','汇总帧级 AP','视频内 macro AUROC','视频级 Max AP']
    point=lambda p:{k:v['value'] for k,v in read(p)['metrics'].items()}
    rows=[('原生零样本',final['metrics']['baseline']),('旧抽样 top-2 MIL',point('results/study_formal_v2/qwen_formal_mil_p1_4s.json')),
          ('教师约束 V2，第1轮',read('results/qwen_localization_v2_epoch1_test/summary.json')['metrics']['candidate']),
          ('零样本＋拼图',read('results/storyboard_context_formal_v1/summary.json')['metrics']['storyboard']),
          ('min-max软标签，第2轮',read('results/qwen_localization_v4_epoch2_test/summary.json')['metrics']['candidate']),
          ('min-max软标签，第5轮',read('results/qwen_localization_v4_epoch5_test/summary.json')['metrics']['candidate']),
          ('min-max软标签，第9轮（平台期后最终模型）',final['metrics']['candidate'])]
    lines=['# Qwen 后续研究：平台期模型完整测试与结论证据','',f'更新：{datetime.datetime.now(datetime.timezone.utc).isoformat()}。提案13个必做条件及5个追加正式预测条件均已完成；范围为实验、分析和结论材料，不含海报制作。','',
        '最终第9轮模型在完整测试集上的汇总帧级AUROC为86.69%、AP为41.30%，视频内AUROC为76.67%，视频级Max AP为94.14%。相对零样本，帧级AUROC/AP和视频级AP的名义95%配对差值区间高于零；视频内AUROC提高0.90个百分点，但区间含零，尚不能证明定位稳定改善或严格不退化。','',
        '相对第5轮，继续至平台期支持汇总帧级AUROC的额外收益；另外三项直接差值区间含零。相对旧MIL，汇总帧级AUROC和视频内AUROC更高，名义区间支持这两项差异；帧级AP虽有较大正点估计，但区间仍含零。视频级AP点估计低约1.48个百分点，直接区间含零；这不能证明等效或非劣效。','',
        '## 1. 正式结果及配对不确定性','',
        '下表均覆盖固定290个测试视频、37188窗口、1111808帧。Qwen使用P1、4秒窗口、1秒步长；逐帧分数取覆盖窗口的原始logit均值，视频级AP固定取窗口最大值。视频内macro对140个同时含异常和背景帧的视频等权平均。单位为百分比。','',
        '|条件|'+'|'.join(labels)+'|','|---|---:|---:|---:|---:|']
    for label,row in rows:lines.append('|'+label+'|'+'|'.join(f'{100*row[k]:.2f}' for k in keys)+'|')
    lines+=['','原13条件的模型、提示词、窗口长度及规模对照保留于[主矩阵](../study_formal_v2/README.md)，机器清单见[inventory.json](inventory.json)。第2/5/9轮属于同一训练轨迹，不能当成独立种子复现。','',
        '![统一检测与定位点估计](frame_transfer.png)','']
    def table(title,values):
        lines.extend(['### '+title,'','|指标|差值（百分点）|95%视频配对区间（百分点）|','|---|---:|---|'])
        for k,label in zip(keys,labels):
            d=values[k];ci=d['ci95'];lines.append(f'|{label}|{100*d["difference"]:+.2f}|[{100*ci["low"]:+.2f}, {100*ci["high"]:+.2f}]|')
        lines.append('')
    table('第9轮减零样本',final['paired_contrasts'])
    lines+=['![第9轮相对零样本的配对区间](../qwen_localization_v4_plateau_test/paired_differences.png)','']
    table('第9轮减第5轮：继续训练的增益',direct['contrasts']['minmax_soft_minus_minmax_soft_epoch5'])
    table('第9轮减旧MIL',direct['contrasts']['minmax_soft_minus_old_mil'])
    lines+=['按视频配对、成簇重采样1000次，seed42；保持同一视频在不同模型下共同抽中。主指标区间仅描述固定预测在视频重采样下的不确定性，不包含重训随机性或标注不确定性。直接方法间比较为冻结模型后的描述性比较，未做多重比较校正；没有预注册非劣效界限，也没有进行等效/非劣效检验。区间含零不能证明无差异或不退化。','',
        '第9轮相对第2轮的帧级AUROC还高约2.77个百分点，直接区间约[+0.36,+5.79]；其余三项区间含零。教师约束模型的完整比较也保留于[method_comparisons.json](../qwen_localization_v4_plateau_test/method_comparisons.json)，不把接近零的区间下界四舍五入成已确认收益。','',
        '## 2. 视频分类收益转移到了哪些帧','',
        '共有84338个异常帧、648905个正常视频帧和378565个异常视频背景帧。后两组负帧的数量权重为63.16%/36.84%；所有异常帧分别与两组负帧比较，再按此权重相加，可精确重建汇总AUROC。异常视频背景比较包含跨视频配对，不能等同于视频内macro AUROC。','',
        '|条件|异常帧 对 正常视频帧 AUROC|异常帧 对 异常视频背景 AUROC|正常视频FPR@约80%异常召回|异常背景FPR@同召回|','|---|---:|---:|---:|---:|']
    for name in ['zero_shot','old_mil','minmax_soft','minmax_soft_epoch5',arm]:
        r=transfer['results'][name];g=r['score_groups'];lines.append(f'|{name}|{100*r["anomaly_vs_normal_video_auroc"]:.2f}|{100*r["anomaly_vs_abnormal_background_auroc"]:.2f}|{100*g["normal_video"]["false_positive_rate_at_80pct_anomaly_recall"]:.2f}%|{100*g["abnormal_background"]["false_positive_rate_at_80pct_anomaly_recall"]:.2f}%|')
    c=read('results/future_work_v1/auc_contributions.json')['contributions'][arm]
    lines+=['',f'第9轮的汇总AUROC增益精确分解为：正常视频比较贡献{c["normal_video_comparison_pp"]:+.2f}个百分点、异常视频背景比较贡献{c["abnormal_background_comparison_pp"]:+.2f}，合计+6.34。收益仍主要来自正常视频与异常帧的分离。异常背景比较AUROC点估计略升，但同召回下的异常背景误报率仍略高于零样本，不能说所有定位角度都改善。','',
        '同视频内部正负帧对仅占全部正负帧对约0.3879%，所以汇总AUROC很容易受到跨视频分数差异影响。按帧对数加权的视频内AUROC与按视频等权的macro指标也不是同一个指标。该数学分解是描述证据，不是某个损失项、抽样策略或奖励机制的因果证明。','',
        '80%召回阈值只作测试后描述，不用于训练、选checkpoint或部署校准；固定logit零阈值的召回与误报可一起上升，不能仅凭绝对分数均值判断排序质量。分位数及其他工作点保留于[完整分解](frame_transfer_analysis.json)。','',
        '![汇总AUROC贡献分解](frame_auc_decomposition.png)','',
        '## 3. 定位失败、分层和固定案例','']
    values=[r['within_deltas'][arm] for r in pv];up=sum(x>1e-12 for x in values);down=sum(x< -1e-12 for x in values)
    lines+=[f'140个混合异常视频中，第9轮相对零样本有{up}个提高、{down}个下降、{len(values)-up-down}个基本不变（绝对差≤1e-12）。平均点估计为正并不表示每个视频都保持。','',
        '|分层|视频数|视频内AUROC变化（百分点）|探索性95%区间（百分点）|','|---|---:|---:|---|']
    for field in ['fraction_bin','duration_bin']:
        for label,r in transfer['stratified_within_differences'][field].items():
            ci=r['exploratory_ci95'][arm];name=('异常占比 ' if field=='fraction_bin' else '时长 ')+label
            lines.append(f'|{name}|{r["videos"]}|{100*r["mean_within_deltas"][arm]:+.2f}|[{100*ci["low"]:+.2f}, {100*ci["high"]:+.2f}]|')
    lines+=['','异常占比10–30%组（n=60）的名义区间高于零；其余占比分组和三个时长组区间均含零。短异常（≤10%，n=50）点估计仍略低于零样本；不能据此确认稳定退化，也不能宣称短异常问题已解决。类别分层的全部样本量、点估计和区间在JSON中保留；分层未经多重校正，不作机制或因果结论。','',
        '按第9轮减零样本的视频内AUROC差值，固定选最小、中位、最大三例，文件名打破并列。案例只作解释，不再调整训练、模型或测试标签。','',
        '![第9轮固定案例曲线](../qwen_localization_v4_plateau_test/case_curves_shape.png)','',
        '曲线逐视频min-max仅用于显示；所有正式指标使用原始logit。[原始曲线](../qwen_localization_v4_plateau_test/case_curves_raw.png)保留。红带为原基准异常区间。','',
        '|案例|视频|时长秒|异常占比|视频内AUROC变化（百分点）|','|---|---|---:|---:|---:|']
    cases=read(result_dir+'/case_selection.json')['cases']
    for role,label in [('worst','最大下降'),('median','中位变化'),('best','最大改善')]:
        r=cases[role];lines.append(f'|{label}|{r["name"]}|{r["duration_s"]:.2f}|{100*r["anomaly_fraction"]:.2f}%|{100*r["within_delta"]:+.2f}|')
    lines+=['',
        '- [Explosion027画面](../qwen_localization_v4_plateau_test/case_frames/Explosion027_x264.mp4.jpg)：店内座椅附近短暂白光/烟雾，随后人员起身移动。多个模型的高峰落在异常之后，第9轮在约15–17秒背景仍有很高的相对分数，对短异常排序不利；不据此判断具体器具或起因。',
        '- [Explosion043画面](../qwen_localization_v4_plateau_test/case_frames/Explosion043_x264.mp4.jpg)：长街景中约149秒出现强光，之后有人聚集；异常占比仅约1.84%。原生及软标签曲线整体形状接近，对应中位变化约+0.32个百分点。',
        '- [Arson011画面](../qwen_localization_v4_plateau_test/case_frames/Arson011_x264.mp4.jpg)：电梯内人员及自行车，随后出现强光/火焰和遮挡；嵌入时间戳有跳变、重复片段迹象。第9轮视频内排序改善约24.42个百分点，但部分已标注异常区间仍低分，不能解释成每个异常片段都识别正确，也不推断起火原因。','',
        '画面为远端224×224、2fps规范化模型输入的少量带时间戳联系表，原片和权重保留远端；Codex已逐图检查3张联系表，不把它称为原生完整视频审阅。全部源/规范化/图片SHA及抽帧规则见[manifest](../qwen_localization_v4_plateau_test/case_frames/manifest.json)。旧第2/5轮案例继续保留。','',
        '## 4. 训练、平台期和测试冻结','',
        '单个Qwen3.5-2B从原生权重初始化rank8/alpha16、dropout0 LoRA；P1、4秒窗口、1秒步长、BF16、AdamW学习率1e-5，每步8视频。异常视频完整缓存教师sigmoid分数逐视频min-max为软标签；正常目标0，窗口独立BCE。没有focal、Huber、排名损失或时间平滑；极小教师分数跨度的异常视频按固定规则跳过，不复现完整MIST。','',
        '1256训练视频每轮157步；每视频均匀抽25%且最多64个窗口，目标及视频/类别归一化在抽样前固定，按包含概率修正。单窗口分数梯度由其固定目标和权重决定；共享参数、总梯度及统一裁剪仍影响整体更新。1419个训练/验证视频的教师分数全部复用，未刷新。时间标注不进训练；112个帧级开发视频包含34个助手审核近似时间标注的异常视频和78个正常视频，视频AP使用完整163验证视频。','',
        '|完整轮次|开发帧级AUROC|开发帧级AP|开发视频内AUROC|验证视频Max AP|','|---|---:|---:|---:|---:|']
    for epoch in range(2,10):
        source='localization_v4_dev' if epoch==2 else 'localization_v4_continuation_dev' if epoch<=5 else 'localization_v4_plateau_dev'
        r=read(f'results/{source}/epoch_{epoch}_comparison.json')['trained'];lines.append('|'+str(epoch)+'|'+'|'.join(f'{100*r[k]:.2f}' for k in ['dev_pooled_auroc','dev_pooled_ap','within_macro_auroc','video_max_ap'])+'|')
    lines+=['',
        '这是一条被预算/用户停止多次中断、每次完整恢复AdamW/RNG和游标的训练轨迹。第2轮曾因预算停止；第5轮曾因用户看到视频内AUROC连续下降而停止并做正式测试。用户随后明确重新授权继续至原开发帧级AUROC平台期；本次从806步接续，没有重置优化器或改变损失/抽样/验证。','',
        '停止规则为连续两轮没有至少0.2个百分点的合格开发帧级AUROC改善；视频内AUROC、帧级AP、163视频AP不得低于原生零样本。第8/9轮增幅约0.1710/0.0819个百分点，bad_epochs达到2；第9轮仍是最高合格主指标，故选第9轮，而不是退回第7轮。共9个完整轮次、1413次实际更新，从806步新增607次，无未验证第10轮尾部。这里的“收敛”指满足配置的开发平台期，不是数学收敛证明。','',
        f'第9轮测试前冻结的文件SHA为 `{selected["checkpoint_file_sha256"]}`，张量SHA为 `{selected["checkpoint_tensor_sha256"]}`。停止重放、实际载荷、时间和GPU释放见[训练验收](../../research/localization_v4_plateau/complete_training_acceptance.json)。第6至9轮活动时间约1.73/1.63/1.63/1.63小时，保守墙钟上界均小于3小时。只测试冻结的第9轮，未测试第3/4/6/7/8轮，不根据新测试重选。','',
        '历史V3硬候选方案在73.90%/81.92%的工程纯度检查下未达到80%/90%，故未训练；门槛不是论文通用标准。原始失败链保留：教师缓存修复、上传包超限、一次冷启动工作目录/进程丢失、CPU打包75.6GB缓存并逐字节校验恢复，以及SSL空读误报修复。缓存准备只改变I/O；未重算视觉/教师、改变科学训练源或放宽校验。','',
        '## 5. 支持与尚不能支持的结论','',
        '- 支持：相对原生零样本，最终模型提高汇总帧级AUROC/AP和视频级AP；相对第5轮，进一步提高主指标帧级AUROC。',
        '- 支持：相对旧MIL，最终模型的汇总帧级AUROC及视频内AUROC更高，名义配对区间支持这两项差异；主要全局收益仍来自正常视频与异常帧的区分。',
        '- 尚不能支持：相对零样本定位严格不退化或稳定提升、每个视频都改善、视频分类与旧MIL等效、第9轮的AP/视频内/视频AP稳定优于第5轮，以及某个单独损失项或抽样改动的因果作用。',
        '- 本研究没有多种子或消融，不声称SOTA；新方法作为整体接受检验。训练及唯一最终测试完成，不为追求正结果再增加训练路线或测试checkpoint。','',
        '## 6. 复现、成本、资源和AI审核','',
        '以下入口使用已有本地预测，只用CPU，不分配GPU；离线复算已实际运行并与首次自动评分完全一致，见[复算审计](../../research/localization_v4_plateau_test/offline_recomputation_audit.json)。','',
        '```bash',f'PYTHONPATH=src .venv/bin/python scripts/finalize_localization_test.py {receipt_path} --offline',
        'PYTHONPATH=src .venv/bin/python scripts/compare_frozen_localization_methods.py --epoch 9',
        'PYTHONPATH=src .venv/bin/python scripts/analyze_frame_transfer.py --include-teacher --include-soft --include-final --include-plateau',
        'PYTHONPATH=src .venv/bin/python scripts/render_final_localization_cases.py --epoch 9',
        'PYTHONPATH=src .venv/bin/python scripts/refresh_experiment_inventory.py',
        'PYTHONPATH=src .venv/bin/python scripts/render_followup_handoff.py','```','',
        f'最后一次追加训练（含失败、恢复和CPU缓存准备）约{train["additional_training_used_cu"]:.4f}/20 CU；本次第9轮正式测试约{amount(receipt):.4f}/3 CU。定位/拼图阶段合计约{budget["total_estimate_cu"]:.4f}/100 CU，未用约{budget["remaining_cu"]:.4f} CU。计价为报价乘观测时间，不是账号余额或精确逐笔账单，也不涵盖更早全部研究开支。未用训练授权已关闭；全部Colab资源已释放，事件监控已收尾。','',
        '数据与模型配置、缓存准入、checkpoint SHA、源码归档及分配/释放收据见[测试交接](../../research/localization_v4_plateau_test/README.md)，全部预测SHA见[清单](../qwen_localization_v4_plateau_test/prediction_sha256.json)，综合[验收](../../research/future_work_v1/final_acceptance.json)可复核。','',
        'AI用于编码、调试、统计、结论草稿和联系表审核；Hydrogen生成12个新增开发标注草稿，Codex逐条审核后形成34个近似异常时间标注，不是官方逐帧真值或人工独立双标。第9轮案例未修改测试标签，也未影响训练或选模。原[AI记录](../../research/localization_v4/ai_use_and_review.json)与[第9轮补充记录](../../research/localization_v4_plateau_test/ai_use_and_review.json)保留；报告作者仍须独立核对数字、来源和课程披露。','',
        '课程正文6–10页，最终ZIP截止12月11日23:59，时区以Canvas为准；交付范围不含海报，原提案保持。课程来源与原件SHA见[契约](../../research/future_work_v1/course_contract.json)。']
    (dest/'README.md').write_text('\n'.join(lines)+'\n')
    short=['# 平台期第9轮完整测试','', '|指标|零样本|第5轮|最终第9轮|','|---|---:|---:|---:|']
    early=read('results/qwen_localization_v4_epoch5_test/summary.json')['metrics']['candidate']
    for k,label in zip(keys,labels):short.append(f'|{label}|{100*final["metrics"]["baseline"][k]:.2f}%|{100*early[k]:.2f}%|{100*final["metrics"]["candidate"][k]:.2f}%|')
    short+=['','固定第9轮，290视频/37188窗口/1111808帧，零优化器更新，GPU已释放。相对第5轮的帧级AUROC差值+1.52个百分点、95%区间[+0.27,+3.23]；另外三项区间含零。相对原生零样本的定位严格不退化尚未证明。','',
        '完整[摘要](summary.json)、[直接比较](method_comparisons.json)、[统一结论及复现](../future_work_v1/README.md)均已落盘。']
    (ROOT/result_dir/'README.md').write_text('\n'.join(short)+'\n')
    print('Rendered complete epoch9 results and evidence handoff')


if __name__=='__main__':main()
