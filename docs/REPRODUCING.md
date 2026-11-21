# 复现层级与边界

## 已支持：从保存的窗口分数复核结论

参照根目录README安装`sites/research-handoff/dist/requirements.txt`。独立校验器只读取冻结结果，不访问账户，不分配GPU。

- `--verify-files`校验公开资料的SHA-256；隐私清理后的文件清单与原档案来源哈希是两个不同口径。
- `--all --bootstrap 0`重算20条件点估计。
- `--condition qwen_localization_v4_plateau_test --bootstrap 1000 --paired qwen_zs_p1_4s`重算最终模型及配对区间。

消融复算命令（在 `sites/research-handoff/dist` 内运行）

```sh
python verify_results.py --condition qwen_e9_ablation_a_test --bootstrap 1000 --paired qwen_localization_v4_plateau_test
python verify_results.py --condition qwen_e9_ablation_b_test --bootstrap 1000 --paired qwen_localization_v4_plateau_test
```

两条命令用同一290个测试视频进行1,000次配对 bootstrap（seed 42），验证点估计和已登记区间。正向差值表示该消融指标高于 E9。分解数据、定义与抽样哈希见 [消融结果](../sites/research-handoff/dist/data/analysis/e9_ablation.json) 和 [实验协议](../sites/research-handoff/dist/data/analysis/e9_ablation_protocol.json)。

逐帧分数取覆盖该帧的窗口原始logit均值；标签使用1-based inclusive边界，截到实际视频帧数。视频内Macro仅使用固定140个同时含正常及异常帧的视频。详见`data/data_dictionary.md`和`data/protocol.json`。

## 已提供实现，需自备输入：重新推理和训练

推理/训练实现位于`src/`、`research/`和`scripts/`，原研究的科学设置、教师分数和数据划分在`sites/research-handoff/dist/data/training/`。冻结软标签：异常视频教师sigmoid分数逐视频min–max，正常目标0，窗口独立BCE，固定视频/类别归一化及包含概率修正；E2/E5/E9为同一轨迹。

如需重新训练消融，沿用原版 Qwen、教师缓存和窗口独立 BCE。在原版 V4 科学配置基础上，将 `training.max_epochs` 设为 9，并配置 `e9_ablation.target_mode` 为 `permuted_minmax`（A）或 `raw_probability`（B），`e9_ablation.initial_tensor_sha256` 使用原版首步记录的初始张量哈希 `cbf060b49f1d11b6bb8ac0a811f06690487bb5b6fbf8485d4c253fd33593f19e`。A 使用按视频名和 seed 42 派生的独立置换；B 保持原始 sigmoid 概率。两组的有效视频、窗口索引和权重对齐，按固定 1,413 步取检查点。实现见 [软标签构造](../src/openjev_dss/localization_soft.py) 和 [训练循环](../src/openjev_dss/localization_training.py)。

完整GPU复现需要自行准备UCF-Crime官方数据、Qwen/Open-Jev对应revision、匹配的模型运行依赖、预处理/缓存及云端授权。源码包含原工作流的完整性门槛和来源哈希，不能通过删除这些门槛把不匹配的输入冒充原实验。公开代码没有复制原始Colab镜像或私有部署资料，因此不保证在任意新环境无需配置即可重新训练。

私人云盘根目录和quota project改为明确占位值；执行编排前必须换成自己的配置。原特定机器上的Colab解释器路径改为`/opt/colab-cli/bin/python`占位路径。禁止把自己的凭据、收据、运行日志、原视频或模型权重提交进Git。无需任何GPU任务即可完成本项目的结果复核。

## 发布检验

187项CPU回归测试通过；独立20条件评分检查、隐私模式扫描和公开文件清单校验另行记录在`publication_checks.json`。这些检查不构成新推理或训练结果。
