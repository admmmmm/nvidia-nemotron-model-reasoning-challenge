# NVIDIA Nemotron Project Plan

## 1. 项目目标

这个项目的目标不是先在本地硬跑 Nemotron-3-Nano-30B，而是先把一条可迁移、可复用、可提交的完整工程链路搭起来：

- 本地完成数据读取、清洗、切分与格式化
- 本地完成小模型 LoRA / QLoRA 训练与验证
- 本地完成答案抽取、近似判分和提交打包脚本
- 本地完成 synthetic data v1 的生产流程
- 再把同一套数据产物和训练逻辑迁移到 Kaggle，训练正式 Nemotron LoRA adapter

一句话版本：

先把工程跑通，再把模型做强。

## 2. 当前状态

当前仓库已经不是只有原始数据，而是已经有了第一版可继续扩展的工程骨架。

可以把当前状态按 4 个文件夹来理解。

### `data/`

这是“数据资产层”。

它不放核心代码，主要放数据中间产物，包括：

- `train.csv`
  - Kaggle 原始训练集
- `test.csv`
  - Kaggle 原始测试集
- `data/splits/default/`
  - 本地验证切分结果
  - 当前已经生成 `train.jsonl` 和 `val.jsonl`
  - 当前切分规模是 `9025 / 475`
- `data/interim/`
  - 预留给后续清洗、标准化后的中间数据
- `data/synthetic/`
  - 预留给后续 synthetic data

当前数据策略已经确定：

- 原始比赛数据继续保留 `csv`
- 本地切分和训练中间产物优先使用 `jsonl`
- 预测和提交表格继续使用 `csv`

### `src/`

这是“代码逻辑层”，也是当前工程的核心。

里面现在已经拆成了 4 个职责子目录。

#### `src/data/`

负责数据读取、校验、切分和训练格式转换。

- `load.py`
  - 读取 `train.csv / test.csv`
  - 校验 schema
  - 把原始行转成结构化样本
- `preprocess.py`
  - 负责本地 train/val 切分
  - 当前已经实际跑过，并生成默认切分
- `format_sft.py`
  - 负责把原始样本转成训练样本格式
  - 统一 system / user / assistant 结构
  - 统一最终答案输出规则

#### `src/eval/`

负责答案抽取、本地判分、验证和生成预测。

- `answer_extract.py`
  - 从模型输出里抽最终答案
  - 优先抽 `\boxed{}`
  - 兼容数字答案和部分边界情况
- `metric_local.py`
  - 实现本地判分逻辑
  - 支持字符串精确匹配和数值近似匹配
- `validation.py`
  - 用本地 `val` 集评估预测结果
  - 产出验证分数和错误样例
- `predict_local.py`
  - 用本地模型或 LoRA adapter 生成 `id,prediction`
  - 作为本地验证和后续提交前 smoke test 的入口

#### `src/train/`

负责本地小模型 LoRA / QLoRA 训练。

- `lora_utils.py`
  - 放 LoRA 配置和 target module 选择逻辑
- `sft_local.py`
  - 第一版本地训练脚本
  - 当前默认目标是先跑通 `Qwen2.5-*B-Instruct` 的 QLoRA baseline

#### `src/submit/`

负责提交产物打包。

- `package_lora.py`
  - 检查 adapter 目录是否包含必要文件
  - 打包生成最终提交需要的 zip

### `outputs/`

这是“运行结果层”。

它不放原始数据，也不放核心代码，只放跑出来的结果。

当前已经建好的目录有：

- `outputs/adapters/`
  - 预留给 LoRA / QLoRA 训练产物
- `outputs/logs/`
  - 当前已经实际使用
  - 放过验证预测文件和错误样例
- `outputs/submissions/`
  - 预留给最终 `submission.zip`

### `scripts/`

这是“薄入口层”。

这里放的是小工具和辅助脚本，不承载主要业务逻辑。

- `scripts/token_count.py`
  - 计算不同 tokenizer 下的 token 数
  - 支持 OpenAI tokenizer 和 Nemotron 对应的 Hugging Face tokenizer
  - 已经用来验证 `test_prompt.txt` 的真实 token 长度

### 其他顶层文件

- `README.md`
  - 项目启动说明
- `PROJECT_PLAN.md`
  - 当前这份项目计划文档
- `test_prompt.txt`
  - 官方模型的一段长 CoT 输出样本
  - 用于分析风格和 token 预算，不作为比赛 prompt 直接使用

当前我们已经确认的关键信息：

- 你的本地环境是 `RTX 4060 8GB`
- 计划使用 Kaggle 免费额度训练正式提交版本
- `test_prompt.txt` 是官方模型的一段长 CoT 输出，不应直接当比赛 prompt 使用
- 超长 CoT 对分析官方风格有价值，但比赛里必须优先考虑 token 预算和稳定输出
- ???????????gold ????????`val` ???? `1.0000 (475/475)`??????????????
- ??? Qwen baseline ?????????????`Qwen2.5-1.5B-Instruct + QLoRA + 1 epoch + max_length=512` ??? `val` ???? `0.4589 (218/475)`

## 3. 成功标准

我建议把成功标准拆成 4 级。

### Level 1: 工程闭环

- 可以读取 `train.csv / test.csv`
    - 在`src/data/`文件夹下。有`format_sft.py,load.py,preprocess.py`
- 可以构造本地验证集
    - 已生成`data/splits/default/train.jsonl`和`data/splits/default/val.jsonl`。当前切分结果是`9025 / 475`
- 可以训练一个小模型 LoRA baseline
    - ?`src/train/`??????`lora_utils.py,sft_local.py`??? QLoRA ???????`qwen_baseline` adapter ???? `outputs/adapters/qwen_baseline/`
- 可以在本地完成答案抽取和 metric 计算
    - 在`src/eval/`文件夹下。有`answer_extract.py,metric_local.py,validation.py,predict_local.py`
    - ???????????gold ????????`val` ???? `1.0000 (475/475)`??? baseline ??????????? `val` ???? `0.4589 (218/475)`

### Level 2: 可迁移闭环

- 本地训练数据格式和 Kaggle 训练格式一致
- LoRA 导出产物结构清晰
- 可以自动打包出符合比赛要求的提交文件

### Level 3: 数据闭环

- 可以稳定生成 synthetic data
- 可以对 synthetic data 做质检、过滤和版本管理
- synthetic data 在本地验证集上比纯原始训练集更好

### Level 4: 比赛闭环

- 在 Kaggle 上训练 Nemotron 正式 adapter
- 成功提交第一个 `submission.zip`
- 建立 leaderboard 迭代节奏

## 4. 项目原则

为了避免项目一开始就变得很散，我建议我们按这几个原则推进：

- 先做短链路 baseline，不先追求复杂方法
- 先保证答案格式稳定，再追求更长推理
- 所有中间产物都可复现，不做一次性脚本
- 本地与 Kaggle 尽量共用同一套数据 schema 和训练接口
- 每个阶段都必须留出可验证的产物，而不是只留下想法

## 5. 主线工作流

这个项目建议拆成 5 条主线并行推进。

### A. 数据与评测

目标：

- 搞清楚原始数据分布
- 复刻官方答案抽取和本地 metric
- 建立 holdout 验证集

需要产出：

- 数据 schema 文档
- 本地验证切分文件
- `answer_extract` 和 `metric_local` 脚本

### B. 本地替身模型训练

目标：

- 用 1.5B 到 3B 级模型跑通 QLoRA / LoRA
- 验证训练格式、推理格式、导出格式是否合理

建议模型优先级：

1. `Qwen2.5-3B-Instruct`
2. `Qwen2.5-1.5B-Instruct`
3. 其他轻量 instruct 模型作为回退

需要产出：

- 训练脚本
- 推理脚本
- 基线结果记录

### C. Synthetic Data

目标：

- 把数据工程做成项目竞争力，而不是只靠模型本体

第一版建议只做 3 类：

- 答案格式对齐数据
- prompt 改写数据
- teacher 生成的短推理轨迹数据

需要产出：

- synthetic schema
- 数据生成脚本
- 数据过滤脚本
- 数据版本记录

### D. Kaggle 正式训练与提交

目标：

- 把本地工程迁移到 Kaggle 免费额度
- 跑出 Nemotron 正式 LoRA adapter
- 完成第一版提交

需要产出：

- Kaggle notebook
- adapter 导出逻辑
- `submission.zip` 打包脚本

### E. 实验记录

目标：

- 不让实验结果散落在聊天记录和临时文件里

最低要求：

- 每次实验记录模型、数据版本、训练配置、验证结果、备注
- 每个 synthetic 数据版本有来源说明

## 6. 分阶段计划

## Phase 0: 项目启动

目标：

- 完成仓库骨架
- 确认数据字段
- 确认评测逻辑

任务：

- 创建 `src/` 结构
- 创建 `data/`、`outputs/` 目录
- 编写数据加载器
- 编写训练样本格式转换器
- 编写本地答案抽取器和 metric

完成标志：

- 不训练模型，也能跑完整个“读取数据 -> 生成预测文件 -> 本地判分”的空链路

## Phase 1: 本地 baseline

目标：

- 用小模型跑出第一版可用 baseline

任务：

- 建立训练/验证切分
- 统一 prompt-template
- 小模型 QLoRA 训练
- 本地验证推理
- 记录错误样例

完成标志：

- 能稳定输出 `\boxed{answer}`
- 本地验证集上有可重复基线分数

## Phase 2: Synthetic Data v1

目标：

- 用数据增强提升 baseline

任务：

- 设计 2 到 3 种 synthetic 策略
- 生成第一批 synthetic 样本
- 过滤低质量样本
- 做 ablation：原始数据 vs 原始 + synthetic

完成标志：

- synthetic 数据相对 baseline 有明确收益，哪怕收益不大

## Phase 3: Kaggle Nemotron 迁移

目标：

- 把本地工程迁移到正式模型

任务：

- 编写 Kaggle 训练 notebook
- 对齐 Nemotron tokenizer / 模板 / 导出方式
- 训练正式 adapter
- 完成第一次提交

完成标志：

- 成功生成并提交合法 `submission.zip`

## Phase 4: 迭代优化

目标：

- 稳定提升 leaderboard 表现

任务：

- 优化训练数据配比
- 优化推理格式
- 控制 reasoning 长度
- 对高频错误模式定向增强

完成标志：

- 建立固定迭代节奏，每轮实验都能归因

## 7. 首周执行清单

如果从现在开始，我建议首周只做这些高价值动作。

### Day 1

- 建立项目目录结构
- 写数据加载和 schema 校验
- 抽样查看训练数据

### Day 2

- 实现本地答案抽取
- 实现近似判分逻辑
- 切出本地验证集

### Day 3

- 定义统一训练格式
- 完成小模型训练脚本 v0
- 完成小模型推理脚本 v0

### Day 4

- 跑第一版 QLoRA baseline
- 记录错误样例
- 定位格式错误和抽取错误

### Day 5

- 做 synthetic data v0
- 跑原始 vs synthetic 的小规模对比

### Day 6-7

- 整理 Kaggle 训练 notebook 草稿
- 准备第一版迁移

## 8. 推荐仓库结构

```text
.
|-- README.md
|-- PROJECT_PLAN.md
|-- train.csv
|-- test.csv
|-- test_prompt.txt
|-- scripts/
|   |-- token_count.py
|-- src/
|   |-- config/
|   |-- data/
|   |   |-- load.py
|   |   |-- preprocess.py
|   |   `-- format_sft.py
|   |-- eval/
|   |   |-- answer_extract.py
|   |   |-- metric_local.py
|   |   `-- validation.py
|   |-- synth/
|   |   |-- generate.py
|   |   `-- filter.py
|   |-- train/
|   |   |-- sft_local.py
|   |   `-- lora_utils.py
|   `-- submit/
|       `-- package_lora.py
|-- data/
|   |-- interim/
|   |-- splits/
|   `-- synthetic/
`-- outputs/
    |-- adapters/
    |-- logs/
    `-- submissions/
```

## 9. 关键决策

当前我建议先固定这几个默认决策，避免项目迟迟不开工。

- 本地替身模型：`Qwen2.5-3B-Instruct`
- 第一阶段训练方式：`4bit QLoRA + SFT`
- 第一阶段目标：稳定输出 `\boxed{}`，不追求超长 CoT
- 工程主线：`transformers + datasets + peft + accelerate`
- synthetic v1：先做短推理监督，不做超长 CoT 蒸馏

## 10. 当前风险

- 最大风险不是代码，而是本地验证和官方评测偏差
- 第二风险是 reasoning 太长，导致 token 预算被耗尽
- 第三风险是 synthetic data 噪声过大，把模型带偏
- 第四风险是 8GB 显存下训练不稳定，必须控制 batch 和序列长度

## 11. 现在就开始的第一批任务

如果按“今天立刻开工”的节奏，第一批任务的完成情况如下：

1. 已完成：建立 `src/` 骨架和目录结构
2. 已完成：实现 `train.csv / test.csv` 的读取和 schema 校验
3. 已完成：实现本地 `answer_extract.py` 和 `metric_local.py`
4. 已完成：定义统一训练样本格式，准备第一版本地 baseline

这 4 步已经做完，项目现在已经进入“可以连续推进”的状态。

下一步建议直接做这 3 件事：

1. ???? `Qwen2.5-1.5B-Instruct` ?? QLoRA baseline?????
2. ? `predict_local.py` ??? `val` ??????????????????? `0.4589 (218/475)`?
3. ??????????????? `Qwen2.5-3B-Instruct` ??? synthetic data v0?????
