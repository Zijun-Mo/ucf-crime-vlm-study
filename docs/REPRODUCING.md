# 复现层级与边界

## 已支持：从保存的窗口分数复核结论

参照根目录README安装`sites/research-handoff/dist/requirements.txt`。独立校验器只读取冻结结果，不访问账户，不分配GPU。

- `--verify-files`校验公开资料的SHA-256；隐私清理后的文件清单与原档案来源哈希是两个不同口径。
- `--all --bootstrap 0`重算18条件点估计。
- `--condition qwen_localization_v4_plateau_test --bootstrap 1000 --paired qwen_zs_p1_4s`重算最终模型及配对区间。

逐帧分数取覆盖该帧的窗口原始logit均值；标签使用1-based inclusive边界，截到实际视频帧数。视频内Macro仅使用固定140个同时含正常及异常帧的视频。详见`data/data_dictionary.md`和`data/protocol.json`。

## 已提供实现，需自备输入：重新推理和训练

推理/训练实现位于`src/`、`research/`和`scripts/`，原研究的科学设置、教师分数和数据划分在`sites/research-handoff/dist/data/training/`。冻结软标签：异常视频教师sigmoid分数逐视频min–max，正常目标0，窗口独立BCE，固定视频/类别归一化及包含概率修正；E2/E5/E9为同一轨迹。

完整GPU复现需要自行准备UCF-Crime官方数据、Qwen/Open-Jev对应revision、匹配的模型运行依赖、预处理/缓存及云端授权。源码包含原工作流的完整性门槛和来源哈希，不能通过删除这些门槛把不匹配的输入冒充原实验。公开代码没有复制原始Colab镜像或私有部署资料，因此不保证在任意新环境无需配置即可重新训练。

私人云盘根目录和quota project改为明确占位值；执行编排前必须换成自己的配置。原特定机器上的Colab解释器路径改为`/opt/colab-cli/bin/python`占位路径。禁止把自己的凭据、收据、运行日志、原视频或模型权重提交进Git。无需任何GPU任务即可完成本项目的结果复核。

## 发布检验

187项CPU回归测试通过；独立18条件评分检查、隐私模式扫描和公开文件清单校验另行记录在`publication_checks.json`。这些检查不构成新推理或训练结果。
