"""Render the verified follow-up evidence for the course report author."""
import datetime,json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from openjev_dss.followup_budget import spending

def read(path):return json.loads((ROOT/path).read_text())
def epoch2_snapshot():
    folder=ROOT/'results/future_work_v1';soft=read('results/qwen_localization_v4_epoch2_test/summary.json')
    comparisons=read('results/qwen_localization_v4_epoch2_test/method_comparisons.json')
    transfer=read('results/future_work_v1/frame_transfer_analysis.json');budget=spending(ROOT)
    teacher=read('results/qwen_localization_v2_epoch1_test/summary.json');montage=read('results/storyboard_context_formal_v1/summary.json')
    native=read('results/study_formal_v2/qwen_zs_p1_4s.json');mil=read('results/study_formal_v2/qwen_formal_mil_p1_4s.json')
    training=read('research/localization_v4/result_summary.json');selected=read('research/localization_v4/selected_checkpoint.json')
    dev=read('results/localization_v4_dev/epoch_2_comparison.json');gates=read('research/localization_v3/pseudo_label_quality_gate.json')
    point=lambda r:{k:v['value'] for k,v in r['metrics'].items()}
    keys=['pooled_auroc','pooled_ap','within_macro_auroc','video_max_ap'];labels=['汇总帧级 AUROC','汇总帧级 AP','视频内 macro AUROC','视频级 Max AP']
    rows=[('零样本',point(native)),('旧抽样 top-2 MIL',point(mil)),('教师约束 V2，第1轮',teacher['metrics']['candidate']),('零样本＋拼图',montage['metrics']['storyboard']),('冻结 min-max 软标签 V4，第2轮',soft['metrics']['candidate'])]
    lines=['# Qwen 后续实验：正式结果、分析与报告材料','',f'更新：{datetime.datetime.now(datetime.timezone.utc).isoformat()}。范围为实验、分析、结论与复现证据；海报由报告团队处理。','',
      'V4在完整测试集上提高了帧级AUROC、帧级AP和视频级AP；三项配对区间均高于零。视频内平均AUROC点估计接近零样本，区间跨零，不能证明定位显著改善或严格不退化。与旧MIL相比，视频内排序明显改善，视频分类仍有小幅差距。','',
      '## 1. 统一正式结果','',
      '每个条件均为相同290个测试视频、37188个窗口、1111808帧。Qwen配置为P1、4秒窗口、1秒步长；视频级AP统一使用窗口最大分数。视频内macro指标对140个同时含正常与异常帧的视频等权平均。表内单位为百分比。','',
      '|条件|汇总帧级 AUROC|汇总帧级 AP|视频内 macro AUROC|视频级 Max AP|','|---|---:|---:|---:|---:|']
    for name,value in rows:lines.append('|'+name+'|'+'|'.join(f'{100*value[k]:.2f}' for k in keys)+'|')
    lines += ['', '提案13个必做条件已完成，提示词、窗口长度和规模对照见[主矩阵报告](../study_formal_v2/README.md)及[原验收](../study_formal_v2/FINAL_ACCEPTANCE.md)。拼图、教师约束与V4作为额外正式实验单独保留。','',
      '### V4减零样本：配对区间','', '|指标|差值（百分点）|视频配对95%区间（百分点）|','|---|---:|---|']
    for key,label in zip(keys,labels):
        d=soft['paired_contrasts'][key];ci=d['ci95'];lines.append(f'|{label}|{100*d["difference"]:+.2f}|[{100*ci["low"]:+.2f}, {100*ci["high"]:+.2f}]|')
    lines += ['', '使用固定模型、按视频成簇的1000次配对bootstrap，种子42。它不包含重训随机性、标注重新审核或训练种子变化；未进行非劣效检验，也没有预注册非劣效界限。','',
      '![V4相对零样本的配对区间](../qwen_localization_v4_epoch2_test/paired_differences.png)','',
      '### V4减旧MIL：识别与定位的取舍','', '|指标|差值（百分点）|视频配对95%区间（百分点）|','|---|---:|---|']
    for key,label in zip(keys,labels):
        d=comparisons['contrasts']['minmax_soft_minus_old_mil'][key];ci=d['ci95'];lines.append(f'|{label}|{100*d["difference"]:+.2f}|[{100*ci["low"]:+.2f}, {100*ci["high"]:+.2f}]|')
    lines += ['', '视频级AP距旧MIL约1.84个百分点，区间低于零，不能称为分类能力等效。视频内AUROC比旧MIL高约5.16个百分点，区间高于零。帧级AP高约7.03个百分点只是点估计，其直接比较区间跨零。主指标汇总帧级AUROC比旧MIL低约0.69个百分点，区间也跨零；相近点估计不等于等效。此处方法间比较属于冻结模型后的描述性比较，未做多重比较校正。','',
      '![正式检测与定位结果](frame_transfer.png)','',
      '## 2. 分类收益转移到了哪些帧','',
      '测试集有84338个异常帧、648905个正常视频帧、378565个异常视频背景帧。把所有异常帧分别与后两组负帧比较，按负帧数量加权，可精确重建汇总帧级AUROC；权重分别为63.16%与36.84%。第二组比较仍包含不同异常视频间的配对，不能等同于视频内macro AUROC。','',
      '|条件|异常帧 对 正常视频帧 AUROC|异常帧 对 异常视频背景帧 AUROC|','|---|---:|---:|']
    for name in ['zero_shot','old_mil','teacher_constraint','minmax_soft']:
        r=transfer['results'][name];lines.append(f'|{name}|{100*r["anomaly_vs_normal_video_auroc"]:.2f}|{100*r["anomaly_vs_abnormal_background_auroc"]:.2f}|')
    contrib=read('results/future_work_v1/auc_contributions.json')['contributions']['minmax_soft']
    lines += ['',f'V4的汇总AUROC提升由正常视频比较贡献约 {contrib["normal_video_comparison_pp"]:+.2f} 个百分点、异常视频背景比较贡献约 {contrib["abnormal_background_comparison_pp"]:+.2f} 个百分点组成。收益主要来自正常视频与异常帧的区分，异常视频背景的跨视频校准问题仍存在。这是固定预测的精确分解，不是某个训练机制的因果证据。','',
      '同一视频内部正负帧对只占全部正负配对的约0.3879%；汇总帧级AUROC对跨视频分数差异敏感。按帧对数加权的视频内AUROC与按视频等权的macro AUROC也有不同权重，不能混用。','',
      '![汇总AUROC贡献](frame_auc_decomposition.png)','',
      '### 误报与真实异常分数','',
      '以下在各模型约80%异常帧召回的描述性工作点比较误报率；阈值仅用于解释冻结预测，不进入选模型、主指标计算或部署校准。','',
      '|条件|正常视频误报率|异常视频背景误报率|正常视频平均logit|异常背景平均logit|真实异常平均logit|','|---|---:|---:|---:|---:|---:|']
    for arm in ['zero_shot','old_mil','minmax_soft']:
        group=transfer['results'][arm]['score_groups'];a=group['normal_video'];b=group['abnormal_background'];c=group['true_anomaly']
        lines.append(f'|{arm}|{100*a["false_positive_rate_at_80pct_anomaly_recall"]:.2f}%|{100*b["false_positive_rate_at_80pct_anomaly_recall"]:.2f}%|{a["mean"]:.3f}|{b["mean"]:.3f}|{c["mean"]:.3f}|')
    lines += ['', 'V4相对零样本明显改变了正常视频分数分布；背景误报在相同召回下没有同步改善。相对旧MIL，异常视频背景误报点估计降低。若固定logit零阈值，召回和误报会同时改变；原始均值、分位数及零阈值比例均保存在[分解数据](frame_transfer_analysis.json)，不能只看均值推断误报率。','',
      '## 3. 退化特征与不确定性','',
      '140个混合异常测试视频中，V4相对零样本的视频内AUROC：68个提高、60个下降、12个基本不变（绝对差≤1e-12）。平均接近不变不代表每个视频都保持。','',
      '|分层|视频数|旧MIL减零样本（百分点）|V4减零样本（百分点）|V4探索性95%区间（百分点）|','|---|---:|---:|---:|---|']
    for field in ['fraction_bin','duration_bin']:
        for name,r in transfer['stratified_within_differences'][field].items():
            ci=r['exploratory_ci95']['minmax_soft'];label=('异常占比 ' if field=='fraction_bin' else '时长 ')+name
            interval='样本不足' if ci is None else f'[{100*ci["low"]:+.2f}, {100*ci["high"]:+.2f}]'
            lines.append(f'|{label}|{r["videos"]}|{100*r["mean_within_deltas"]["old_mil"]:+.2f}|{100*r["mean_within_deltas"]["minmax_soft"]:+.2f}|{interval}|')
    lines += ['', '异常占比≤10%的50个视频仍是V4点估计较差的一组，但下降约0.67个百分点，区间跨零；旧MIL同组约下降7.89个百分点。时长分组的V4区间均跨零。类别分层提示Explosion（n=21）与RoadAccidents（n=23）仍有平均下降，完整类别样本量和区间在JSON中；这些探索性分组未校正多重比较，不能据此确定原因。','',
      '## 4. 固定案例、曲线与对应画面','',
      '按V4减零样本的视频内AUROC差值固定选择最小、中位、最大三例，文件名处理并列；选择只用于解释已冻结结果，不用于改模型。','',
      '![V4自身固定案例](minmax_soft_cases_shape.png)','',
      '曲线逐视频归一化仅用于显示排序形状；所有指标均使用原始帧logit。[原始logit曲线](minmax_soft_cases.png)和[旧MIL固定案例的多方法对照](descriptive_cases.png)同时保留。红带为数据集时间标注。','']
    cases=read('results/qwen_localization_v4_epoch2_test/case_selection.json')['cases']
    base={r['name']:r for r in soft['reports']['baseline']['per_video']};cand={r['name']:r for r in soft['reports']['candidate']['per_video']}
    lines += ['|案例规则|视频|时长秒|异常占比|零样本视频内AUROC|V4视频内AUROC|变化百分点|','|---|---|---:|---:|---:|---:|---:|']
    for role,label in [('worst','最大下降'),('median','中位变化'),('best','最大改善')]:
        c=cases[role];name=c['name'];lines.append(f'|{label}|{name}|{c["duration_s"]:.2f}|{100*c["anomaly_fraction"]:.2f}%|{100*base[name]["within_auroc"]:.2f}|{100*cand[name]["within_auroc"]:.2f}|{100*c["within_delta"]:+.2f}|')
    lines += ['', '- [Shoplifting007画面](../qwen_localization_v4_epoch2_test/case_frames/Shoplifting007_x264.mp4.jpg)：顾客、货架与柜台持续共存，标注异常较短；曲线中背景高分说明仍有局部错排，不能从单帧断定具体违法动作。',
      '- [Shooting008画面](../qwen_localization_v4_epoch2_test/case_frames/Shooting008_x264.mp4.jpg)：前段人物快速倒地，后续较长时间保持静止场景；该例中位变化接近零，结合绝对指标解释。',
      '- [Stealing058画面](../qwen_localization_v4_epoch2_test/case_frames/Stealing058_x264.mp4.jpg)：含宽/近景切换和片尾文字，异常标注覆盖约62%；改善约17.30个百分点，不据此推断模型学到了某个具体动作。','',
      '画面来自远端固定224×224、2fps规范化模型输入，均保留时间戳、原始视频SHA与规范化SHA；原片留在远端。取样规则为均匀8点、首尾异常边界附近和两模型最高分窗口中心，SHA清单见[画面清单](../qwen_localization_v4_epoch2_test/case_frames/manifest.json)。Codex已逐图审核；这些低帧率图片不能替代完整视频或新的精确人工标注。','',
      '## 5. 训练、选模型与历史失败记录','',
      'V4从原生Qwen2B重新初始化rank8/alpha16、dropout0的LoRA，P1/4秒/1秒步长、BF16、AdamW学习率1e-5，每步8个视频。教师为同一固定原生零样本模型。异常视频全部窗口的sigmoid教师分数先逐视频min-max成软标签，正常目标为0，使用窗口独立BCE；未加入时间平滑、focal、Huber或额外定位网络。常量分数视频跳过正视频监督。仅借鉴[MIST的软伪标签归一化](https://openaccess.thecvf.com/content/CVPR2021/html/Feng_MIST_Multiple_Instance_Self-Training_Framework_for_Video_Anomaly_Detection_CVPR_2021_paper.html)，不声称复现完整MIST。','',
      '目标与归一化系数在抽样前确定；每视频每轮均匀抽取25%且最多64个窗口，使用固定包含概率修正。固定目标y下，单窗口BCE对分数的导数为p−y乘固定权重；其他抽样窗口不改变该损失项或分数梯度。参数共享、批梯度相加与统一裁剪仍会影响整体更新。1256个训练视频每轮157次更新，验证覆盖163个视频；帧级选模使用34个带近似时间标注的异常视频加78个正常视频。标注只用于验证，未进入训练损失。主指标为112视频帧级AUROC，另外要求视频内AUROC、帧级AP和163视频级AP不低于零样本。','',
      f'实际完成2轮，按开发集选择第2轮；选定检查点对应314次更新。自动循环随后有14步未完成第三轮更新，已排除选模与测试。因15CU训练预算不足完整新轮次而结束，未声称收敛。第1/2轮训练器记录约1.79/1.68小时，含收尾的保守墙钟上界约2.05/1.92小时，满足三小时要求。权重文件SHA为 `{selected["checkpoint_file_sha256"]}`，张量SHA为 `{selected["checkpoint_tensor_sha256"]}`。','',
      '历史V3硬候选/focal方案在预检停止，未训练。其正候选与背景帧纯度约73.90%/81.92%，80%/90%为工程阈值，没有论文通用标准依据。用户随后明确授权V4软标签方案，历史门槛没有被伪装成通过。','',
      '实际故障均保留证据：V1教师缓存仅166份导致重复推理；在零优化器更新时核对996份分数完全一致并补齐423份，随后复用完整1419份。相关支出计入14.70CU训练额度。正式测试实际成功，控制器读取result.json超时误报失败；从已落盘云盘结果核验290份预测后修正结算，没有重新推理或分配GPU。原始失败收据、日志与修正链保留。','',
      '## 6. 支持与不支持的结论','',
      '- 支持：该冻结模型相对零样本提高汇总帧级AUROC/AP和视频级AP；区间支持三项收益。它恢复了大部分旧MIL的视频分类收益，视频内平均排序的点估计接近零样本，且优于旧MIL。',
      '- 支持：全局帧级收益主要来自正常视频与异常帧的分离，异常视频背景仍是薄弱环节。',
      '- 尚不支持：定位严格不退化、所有视频均改善、与旧MIL分类等效、单一损失项或抽样方式的因果贡献、SOTA或多种子稳定性。',
      '- 本阶段不再为追求正结果追加训练路线、种子或消融。提案其余对照问题由已完成主矩阵回答。','',
      '## 7. 复现、成本与AI使用','',
      '以下命令仅重算缓存预测，不占GPU：','', '```bash',
      'PYTHONPATH=src .venv/bin/python scripts/finalize_localization_test.py research/localization_v4_test/20261114T190638Z/receipt.json',
      'PYTHONPATH=src .venv/bin/python scripts/compare_frozen_localization_methods.py',
      'PYTHONPATH=src .venv/bin/python scripts/analyze_frame_transfer.py --include-teacher --include-soft',
      'PYTHONPATH=src .venv/bin/python scripts/refresh_experiment_inventory.py',
      'PYTHONPATH=src .venv/bin/python scripts/render_followup_handoff.py','```','',
      '正式评分口径为原始窗口logit按覆盖帧平均；每视频对齐原始时间轴。零样本只复用缓存。固定策略见[测试policy](../../research/localization_v4_test/policy.json)，云盘预测/所选权重位置、源代码归档、分配报价和清理证据见[收据](../../research/localization_v4_test/20261114T190638Z/receipt.json)；预测SHA、逐视频变化、固定案例与配对区间见[完整测试结果](../qwen_localization_v4_epoch2_test/summary.json)。','',
      f'定位/拼图阶段按报价乘已观测时长估计共 **{budget["total_estimate_cu"]:.4f}/60 CU**，剩约 **{budget["remaining_cu"]:.4f} CU**。V4训练约14.7003/15CU，正式测试约{budget["phases"]["v3_test"]:.4f}/3CU，案例CPU准备约{budget["phases"]["annotation_cpu"]:.4f}CU。这不是账号余额或精确逐笔账单，不包含更早期全部研究开销。Hydrogen额度另记，未混入CU。','',
      'AI用于编码、调试、统计、分析草稿和画面审核。Hydrogen生成12个新增开发标注草稿，Codex逐条结合原生视频呈现、概览拼图与边界图复核；共34个异常开发标注为助手审核的近似区间，不是官方逐帧真值或人工双标。新增案例只辅助解释测试结果，没有修改测试标注或训练目标。见[实际AI使用与审核记录](../../research/localization_v4/ai_use_and_review.json)。报告作者仍需独立核对结论、引用和课程披露。','',
      '课程正文要求6至10页，最终ZIP截止12月11日23:59，时区以Canvas为准；提交PDF报告、代码、README和数据/获取方式。海报制作不在本工作范围。原提案保持原样，课程与原提案哈希见[课程契约](../../research/future_work_v1/course_contract.json)。']
    (folder/'README.md').write_text('\n'.join(lines)+'\n');print('HANDOFF_RENDERED',folder/'README.md')
def main():
    if (ROOT/'results/qwen_localization_v4_plateau_test/summary.json').exists():
        from render_plateau_followup_handoff import main as render_plateau
        return render_plateau()
    if (ROOT/'results/qwen_localization_v4_epoch5_test/summary.json').exists():
        from render_final_followup_handoff import main as render_final
        return render_final()
    return epoch2_snapshot()

if __name__=='__main__':main()
