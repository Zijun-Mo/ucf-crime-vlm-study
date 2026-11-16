# UCF-Crime研究与结果交接

这是已冻结18条件的研究资料，不含源视频、模型权重或视觉特征。无需远端账号或GPU即可从窗口分数复核指标。

## 阅读网页

若目录内含index.html，直接用浏览器打开即可；章节、筛选、提示词复制、分数曲线与单文件下载支持离线。分类ZIP不包含在离线网页中，解压数据直接位于data/。

## CPU复算

需要Python 3.10+，建议独立虚拟环境。安装requirements.txt后运行：

```sh
python verify_results.py --verify-files
python verify_results.py --all --bootstrap 0
python verify_results.py --condition qwen_localization_v4_plateau_test --bootstrap 1000 --paired qwen_zs_p1_4s
```

--bootstrap 0核验点估计，不伪装成统计区间；--bootstrap 1000使用注册的seed42、视频配对抽样权重，核对已有区间。任何原始预测身份、帧覆盖、点估计、已注册区间或SHA错误都会报错退出。

程序不调用模型，不重新推理，不训练，也不进行阈值拟合。没有下载源视频不影响这类“从已保存预测复算”，但它不等于重现从视频到模型分数的完整GPU工作流。

## 数据位置

- data/summary.csv、summary.json：18条件与10项指标；页面主要展示其中6项。
- data/test_videos.json：290视频元数据和原测试时间标注。
- data/predictions/：各条件全部290视频窗口分数，数值不截断。
- data/per_video/：逐视频指标。
- data/statistics/、comparisons.json：已注册及冻结后描述性配对比较。
- data/training/：9轮开发指标、已保存开发预测、1419教师分数、划分和近似时间标注、配置、选模与停止记录。
- data/analysis/、assets/figures/、assets/cases/：分解、分层、固定案例及真实图表。
- data/provenance/：原始报告和预测来源SHA、配置、验收、AI审核与成本口径。
- data/history/：早期不同协议结果，单独保存。

字段与评分口径见data/data_dictionary.md。方法来源见data/references.bib，结论与组员接手内容见data/handoff.md。

全部数据是2026-11-15冻结快照。不要用历史文件中的旧运行状态覆盖当前验收；E2／E5／E9是同一轨迹，不当独立种子。开发时间标注是助手审核的近似区间，不是官方逐帧标签；仅用于验证与分析，不进入训练损失。
