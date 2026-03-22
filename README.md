# NVIDIA Nemotron Model Reasoning Challenge

本项目用于参加 Kaggle 比赛 **NVIDIA Nemotron Model Reasoning Challenge**，目标是在本地先完成可复用的 `数据生成 + LoRA 微调 + 本地评测 + 提交打包` 管线，再把同一套工程思路迁移到 Kaggle 免费额度上跑正式的 Nemotron 提交。

本文档是项目启动说明，基于 2026-03-22 抓取的 Kaggle 比赛元数据、规则页、描述页、评测页、数据说明页、时间线和奖项页整理而成。

## 1. 比赛事实摘要

| 项目 | 信息 |
| --- | --- |
| 比赛名 | NVIDIA Nemotron Model Reasoning Challenge |
| 主办方 | NVIDIA |
| 比赛开始 | 2026-03-16 |
| 截止时间 | 2026-06-15 23:59 UTC |
| 中期节点 | 2026-04-09 |
| 团队上限 | 5 人 |
| 每日提交上限 | 5 次 |
| 最终可选提交数 | 2 个 |
| 公榜占比 | 50% |
| 奖项 | 总计 `$106,388`，含现金和 NVIDIA DGX Spark |
| 是否需要实名/身份验证 | 是 |
| 是否需要接受规则 | 是 |
| 提交文件名 | `submission.zip` |
| 提交内容 | Nemotron-3-Nano-30B 的兼容 LoRA adapter |

## 2. 数据与评测约束

### 数据

- `train.csv`
  - `id`: 样本 ID
  - `prompt`: 题目描述，包含输入输出示例和待求解实例
  - `answer`: 标准答案
- `test.csv`
  - `id`
  - `prompt`
- 官方描述这是一组逻辑推理题，覆盖 bit manipulation、代数方程等规则归纳任务。

### 评测

- 官方基座模型是 [`NVIDIA Nemotron-3-Nano-30B`](https://www.kaggle.com/models/metric/nemotron-3-nano-30b-a3b-bf16)。
- 评测时会用 `vLLM` 加载你的 LoRA adapter。
- 模型被要求把最终答案放进 `\boxed{}`。
- 评测先优先抽取 `\boxed{}` 中答案，失败时再走启发式抽取，最后可能退化到“最后一个数字”。
- 判分规则：
  - 字符串完全匹配算对
  - 或数值在相对误差 `1e-2` 内算对
- 最终分数是准确率 `Accuracy`。

### 官方评测参数

| 参数 | 值 |
| --- | --- |
| `max_lora_rank` | `32` |
| `max_tokens` | `7680` |
| `top_p` | `1.0` |
| `temperature` | `0.0` |
| `max_num_seqs` | `64` |
| `gpu_memory_utilization` | `0.85` |
| `max_model_len` | `8192` |

### 关键提交限制

- 最终提交必须是 **rank 不超过 32** 的 LoRA。
- 压缩包里必须包含 `adapter_config.json`。
- 目标基座必须兼容 **Nemotron-3-Nano-30B**。

## 3. 奖项与合规约束

- 进入奖项评审，需要公开 Kaggle notebook 和 solution write-up。
- 优胜方案需要提供：
  - 训练代码
  - 推理代码
  - 运行环境说明
  - 可复现的方法说明
- 规则允许使用外部数据和外部模型，但要求：
  - 对所有参赛者“合理可获得”
  - 成本不能离谱
  - 不能依赖明显不公平或封闭的资源
- 规则允许：
  - prompting
  - data filtering / curation
  - synthetic data generation
  - reinforcement learning
  - lightweight fine-tuning

## 4. 我们的项目目标

### 主要目标

- 在本地 4060 8GB 环境先完成一套可迁移到 Nemotron 的工程骨架。
- 本地优先把下面四件事做扎实：
  - 数据读取与格式化
  - synthetic data 生成
  - 小模型替身 LoRA 微调
  - 本地评测与提交打包
- 在 Kaggle 免费额度上再切到正式基座，训练真正可提交的 Nemotron LoRA adapter。

### 非目标

- 不在本地强行复现 Nemotron-3-Nano-30B 训练。
- 不在第一阶段就上 RL。
- 不在还没有稳定 baseline 前做复杂多框架迁移。

## 5. 推荐技术路线

### 5.1 总体策略

采用“两段式”路线：

1. 本地用小模型做替身，验证数据与训练管线。
2. Kaggle 上换成 Nemotron-3-Nano-30B，复用同一套数据产物和导出逻辑。

### 5.2 本地替身模型建议

优先建议：

- `Qwen2.5-3B-Instruct` + 4bit QLoRA

备用回退：

- `Qwen2.5-1.5B-Instruct`
- `Llama-3.2-3B-Instruct`

选择理由：

- 3B 级模型在 8GB 显存上更容易把 SFT/QLoRA 管线先跑通。
- 指令模型更方便直接对 `prompt -> final boxed answer` 格式做监督。
- 先把“工程接口一致性”做好，比本地追求极限模型能力更重要。

### 5.3 本地训练框架建议

第一版建议统一采用 Hugging Face 生态：

- `transformers`
- `datasets`
- `peft`
- `trl` 或原生 `Trainer`
- `bitsandbytes`
- `accelerate`

可选增强：

- `unsloth`

理由：

- 本地和 Kaggle 都更容易复用。
- 导出 LoRA adapter 的路径更清晰。
- 后续如果要切到 Axolotl/Unsloth，也能保留数据格式和评测脚本。

### 5.4 训练策略建议

第一阶段只做 `SFT baseline`：

- 输入：原始题目 `prompt`
- 输出：模型回答，强制包含 `\boxed{final_answer}`
- 先不追求长链路 CoT 质量，优先保证：
  - 答案抽取稳定
  - 训练格式统一
  - 本地 metric 能和官方 metric 大致对齐

第二阶段再做数据增强：

- teacher 生成 reasoning traces
- 自一致性筛选
- 规则题模板变换
- prompt 改写和样本清洗

第三阶段才考虑：

- RL / preference optimization
- 更激进的 prompt curriculum
- 多阶段训练

## 6. Synthetic Data 方向

这个比赛非常适合把数据工程做成核心竞争力。建议 synthetic data 从下面几类开始：

### A. 答案格式对齐数据

- 目标：让模型稳定输出 `\boxed{}`
- 做法：把训练集答案包装成统一风格的监督样本

### B. 解题轨迹蒸馏数据

- 用 teacher 模型为训练集生成简洁推理过程
- 只保留能得到正确答案的轨迹
- 可以保留“短链条 + 末尾 boxed answer”的风格

### C. Prompt 改写数据

- 不改答案，只改题目表达方式
- 保持规则不变，提升鲁棒性

### D. 规则变换数据

- 针对可程序化的题型做同分布增广
- 例如：
  - 数值替换
  - 符号重命名
  - 样例顺序扰动
  - 等价表达改写

### E. 负样本过滤数据

- 用 teacher/self-consistency 过滤低质量合成样本
- 先保守，不要一开始追求数据量

## 7. 本地与 Kaggle 的职责划分

### 本地负责

- 数据解析
- dataset schema 统一
- synthetic data 生成
- 小模型 SFT/QLoRA 验证
- 本地 metric 复刻
- adapter 导出逻辑
- `submission.zip` 打包脚本
- 提交前 smoke test

### Kaggle 负责

- 正式 Nemotron-3-Nano-30B LoRA 训练
- 更长上下文和更大 batch 的实验
- 最终 adapter 导出
- leaderboard 迭代

## 8. 建议仓库结构

```text
.
├─ README.md
├─ data/
│  ├─ raw/
│  ├─ interim/
│  ├─ synthetic/
│  └─ splits/
├─ notebooks/
│  ├─ kaggle_train_nemotron.ipynb
│  └─ local_explore.ipynb
├─ src/
│  ├─ config/
│  ├─ data/
│  │  ├─ load.py
│  │  ├─ preprocess.py
│  │  ├─ format_sft.py
│  │  └─ synth_generate.py
│  ├─ train/
│  │  ├─ sft_local.py
│  │  ├─ sft_kaggle.py
│  │  └─ lora_utils.py
│  ├─ eval/
│  │  ├─ answer_extract.py
│  │  ├─ metric_local.py
│  │  └─ validation.py
│  ├─ submit/
│  │  ├─ package_lora.py
│  │  └─ smoke_test.py
│  └─ utils/
├─ outputs/
│  ├─ adapters/
│  ├─ logs/
│  └─ submissions/
└─ scripts/
   ├─ run_local_train.ps1
   ├─ run_local_eval.ps1
   └─ build_submission.ps1
```

## 9. 第一阶段执行清单

建议我们按这个顺序推进：

1. 建 `train.csv / test.csv` 的读取和 schema 校验。
2. 复刻官方答案抽取与近似判分逻辑。
3. 定义统一训练样本格式：
   - `input_text`
   - `target_text`
   - 必须显式输出 `\boxed{}`
4. 用小模型跑通第一版 QLoRA SFT。
5. 写 adapter 导出与 `submission.zip` 打包脚本。
6. 做 synthetic data v0：
   - 格式对齐
   - prompt 改写
   - teacher 轨迹蒸馏
7. 再迁移到 Kaggle 的正式 Nemotron 训练 notebook。

## 10. 近期里程碑

### Milestone 1

本地 baseline 跑通

- 能训练
- 能验证
- 能导出 LoRA
- 能打包

### Milestone 2

synthetic data v0 有收益

- 本地验证集比纯原始数据更好

### Milestone 3

Kaggle 正式 Nemotron 提交打通

- 生成合法 `submission.zip`
- 完成第一次 leaderboard 提交

## 11. 当前风险判断

- 最大风险不是代码，而是 **本地验证和官方评测偏差**。
- 第二个风险是 **Nemotron 的 prompt 格式敏感性**，小模型上有效的输出模板不一定能无缝迁移。
- 第三个风险是 **synthetic data 质量控制**，低质量数据很容易把 baseline 拉垮。
- 第四个风险是 **4060 8GB 显存限制**，所以本地一定要把“工程验证”和“正式训练”分层处理。

## 12. 我建议我们先做的版本

如果要尽快进入可执行状态，我建议第一版就按下面的最小闭环做：

- 本地模型：`Qwen2.5-3B-Instruct` 4bit QLoRA
- 目标任务：只学会从 `prompt` 产出稳定 `\boxed{answer}`
- 数据：原始训练集 + 少量高质量 synthetic
- 训练：单阶段 SFT
- 验证：本地 holdout accuracy
- 导出：标准 LoRA adapter + `submission.zip`

这是最务实、也最适合你当前机器条件的起点。

## 13. 下一步讨论建议

接下来我们最值得先拍板的是三件事：

1. 本地替身模型到底选哪一个。
2. synthetic data 第一版用哪种 teacher。
3. Kaggle 端是继续 Hugging Face + PEFT，还是直接切 Unsloth/Axolotl。

我的默认建议是：

- 本地先用 `Qwen2.5-3B-Instruct`
- 先用 SFT，不先上 RL
- 工程主线先用 Hugging Face + PEFT，后面再评估是否切 Unsloth

## 14. 参考链接

- 比赛首页: <https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/overview>
- 评测页: <https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/overview/evaluation>
- 数据说明页: <https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/data>
- 规则页: <https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/rules>
- 官方 metric 说明页: <https://www.kaggle.com/code/metric/nvidia-nemotron-metric>
- 提交 demo: <https://www.kaggle.com/code/ryanholbrook/nvidia-nemotron-submission-demo>
