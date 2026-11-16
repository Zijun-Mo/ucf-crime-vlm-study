"""Pure completion-message construction, testable without sending a message."""


def completion_prompt(receipt,event,receipt_path):
    formal=receipt.get('formal_test',False)
    parts=[f'[整段视频拼图上下文实验事件] 状态 {receipt["status"]}。证据 {event}；收据 {receipt_path}。',
        '用户授权0-shot Qwen整段均匀抽帧拼图实验。先读结果，核验GPU释放；禁止重复分配。']
    if formal:
        parts.extend([
            '汇报完整290视频正式测试对照的汇总帧级AUROC/AP、视频内AUROC和视频AP。',
            '用户已缩减为只推理文字＋拼图，原始baseline复用冻结分数；不要补跑仅文字组。报告相对baseline的成对置信区间，核验视觉缓存命中及数值门。',
            '核对290视频、1111808帧、37188窗口全部完成；在results/storyboard_context_formal_v1保存正式报告、逐视频结果及成本，核验基线复现。',
            '本正式实验各次分配合计预算11CU，整个已有研究预算40CU仍有效。'])
    else:
        parts.extend([
            '汇报44视频开发对照的汇总帧级AUROC/AP、视频内AUROC和视频AP，以及拼图相对仅加文字的差异和成对置信区间。',
            '结果不等于完整测试集。本实验各次分配合计预算3CU，整个已有研究预算40CU仍有效。'])
    parts.extend(['不得恢复训练或扩展参数扫描。失败时在剩余预算内修复；仅允许L4。',
        '处理后在completion_event.json写handled_utc并结束回合，不启动周期性模型轮询。'])
    return ''.join(parts)
