# Project 1：用 NumPy 复现 OpenCLIP ViT-B/32 图像编码器

> 代码仓库（公开）：https://github.com/gao066/project1-numpy-image-encoder

本项目完成课程 Project 1 的六组实现任务：在不使用 PyTorch / OpenCLIP 的前提下，仅用 NumPy 复现
`szu_python_openclip` 最终项目图像塔的完整前向计算，输出与给定参考在 float32 容差内一致。

- 输入：预处理后的图像批 `[B,3,224,224]`（float32）
- 内部：CLS + 49 个 patch token，共 `[B,50,768]`，经 12 层 Transformer
- 输出：每张图一个单位长度的 `[B,512]` 向量，即最终项目 `ImageEncoder.__call__` 的返回值

## 运行前必读：两个课程数据文件

本仓库已包含课程 Starter 的**全部源码与脚本**（`check.py`、`helpers.py`、`preprocess_image.py`、`runtime.py`、`run.py`、`transformer_steps.py`、`config.json`、`assets/images`、`lecture-demos` 等），
但**未上传**下面两个大文件——课程 README 明确要求不要重复提交参数文件与参考数据，且 312 MB 超过 GitHub 单文件 100 MB 的限制：

| 文件 | 大小 | 用途 |
| --- | --- | --- |
| `weights/image_encoder.npz` | 312 MB | 152 个视觉张量（模型参数） |
| `assets/reference_outputs.npz` | 7.4 MB | 分阶段参考中间结果（仅用于比对） |

**获取方法**：从课程平台下载 `project1-学生版.zip`，解压后把这两个文件分别放回 `weights/` 与 `assets/`，
即可完整运行 `python check.py --stage all` 与 `python run.py --output output_tokens.npz`。

## 目录结构

```
project1-numpy-image-encoder/
├── image_encoder.py        # 本项目的唯一改动文件（六组 TODO 的实现）
├── helpers.py              # 课程提供：LayerNorm、精确 GELU、softmax、单位化
├── preprocess_image.py     # 课程提供：短边缩放、中心裁剪、CLIP 归一化
├── runtime.py / run.py     # 课程提供：资产加载与命令行入口
├── check.py                # 课程提供：分阶段数值检查（参考数组仅在此读取）
├── config.json             # 固定 checkpoint 的架构参数
├── weights/                # 课程提供：152 个视觉张量（312 MB，不纳入仓库）
├── assets/                 # 课程提供：10 张图片、ID 清单与参考中间结果
├── lecture-demos/          # 课程提供：课堂 NumPy 示例
├── transformer_steps.py    # 课程提供：迷你 Transformer 逐式计算演示
├── requirements.txt        # 依赖清单
├── evidence/               # 本项目的验证证据
│   ├── check_all.log       # check.py --stage all 的完整输出
│   ├── extra_verify.py     # 额外鲁棒性测试脚本
│   ├── extra_verify.log    # 上述脚本的输出
│   ├── debug_variants.py   # 两类典型错误的对照实验脚本
│   ├── debug_variants.log  # 两类错误的实测偏差
│   ├── timing.py           # 十张图前向计时脚本
│   └── timing.log          # 计时输出：32.8 s / 32.3 s
├── report/                 # 实验报告：Word 源文件与导出 PDF
└── practice/               # 课程提供的练习材料（project1_exercises.pdf）
```

> 仓库只保留源码、README 与报告。`weights/image_encoder.npz`（312 MB）和
> `assets/reference_outputs.npz`（7.4 MB）体积过大且属于课程资产，请从课程平台下载原始
> `project1-学生版.zip` 解压获得，不要重复上传。

## 环境与依赖

- Python 3.9 及以上（实测 3.13.14）
- numpy >= 1.24, < 3（实测 2.5.3）
- Pillow >= 9.2, < 13（实测 12.3.0）
- 仅 CPU，无需 GPU、PyTorch 或联网

```bash
python -m pip install -r requirements.txt
```

## 运行步骤

在本目录（含 `weights/` 与 `assets/`）下执行：

```bash
# 1) 资产自检：152 个张量的名称/形状/dtype/SHA-256 与参考顺序
python check.py --stage assets

# 2) 按实现顺序逐阶段校验
python check.py --stage patches
python check.py --stage tokens
python check.py --stage block0
python check.py --stage last_tokens
python check.py --stage output
python check.py --stage batch

# 3) 一次性跑完全部阶段
python check.py --stage all

# 4) 对 10 张给定图片生成输出 token
python run.py --output output_tokens.npz

# 5) 也可以对任意图片运行（顺序与输入一致）
python run.py --images path/to/a.jpg path/to/b.png --output my_tokens.npz
```

## 实现要点

| 任务 | 方法 | 形状变化 | 关键操作 |
| --- | --- | --- | --- |
| TODO-01 | `patchify` | `[B,3,224,224]` → `[B,49,3072]` | reshape 出 7×7 块网格 → transpose(0,2,4,1,3,5) → 展平 |
| TODO-02 | `split_heads` | `[B,50,768]` → `[B,12,50,64]` | 特征轴拆成（头, 头内维），头轴前移 |
| TODO-03 | `MultiHeadAttention` | `[B,12,50,64]` → `[B,50,768]` | `qkᵀ/√64`、softmax 加权、拼头、输出投影 |
| TODO-04 | `TransformerBlock` | `[B,50,768]` 不变 | 两次残差 + GELU MLP（768→3072→768） |
| TODO-05 | `embed` | `[B,49,3072]` → `[B,50,768]` | 无偏置块投影、拼 CLS、加位置嵌入 |
| TODO-06 | `__call__` | `[B,50,768]` → `[B,512]` | 取 CLS、post-Norm、`[768,512]` 投影、单位化 |

约定：除 `visual.proj` 为 `[in,out]` 外，线性层权重均为 `[out,in]`，统一左乘 `weight.T`；
patch 投影无偏置；图像塔不含 LayerScale，只有两处残差。

## 验证结果（本机实测）

`python check.py --stage all` 全部 PASS：

| 阶段 | 形状 | max_abs | mean_abs | atol |
| --- | --- | --- | --- | --- |
| preprocess | (10,3,224,224) | 0 | 0 | 2e-6 |
| patches | (2,49,3072) | 0 | 0 | 2e-4 |
| tokens | (2,50,768) | 4.005e-5 | 1.289e-6 | 2e-4 |
| block0 | (2,50,768) | 7.063e-6 | 3.250e-7 | 2e-4 |
| last_tokens | (2,50,768) | 4.768e-5 | 2.696e-6 | 2e-4 |
| output | (10,512) | 5.439e-7 | 8.311e-8 | 2e-5 |
| batch | (3,512) | 4.619e-7 | 7.632e-8 | 2e-5 |

额外测试：单张与整批一致（5.2e-8）、乱序批一致（5.8e-8）、重复图像一致、
奇数尺寸/灰度/RGBA 输入均可运行、输出 L2 范数恒为 1.000000、patch 顺序手工核对 diff=0。

耗时（`evidence/timing.log`，Windows 11 / 纯 CPU / Python 3.13.14 / numpy 2.5.3）：
10 张图前向两次分别为 32.78 s 与 32.32 s（约 3.3 s/张），与报告 4.2 节一致。

## 复现证据

```bash
python check.py --stage all            # 分阶段数值检查（assets + 7 项）
python evidence/extra_verify.py        # 批大小、乱序、重复、奇数尺寸/灰度/RGBA
python evidence/debug_variants.py      # 拼头漏转置、float64 缩放因子的偏差
python evidence/timing.py              # 十张图前向计时
```

上述脚本都以仓库根目录为基准定位文件，换机器只需放回 `weights/` 与 `assets/` 两个课程
数据文件，不依赖绝对路径。

## 参考数据说明

`assets/reference_outputs.npz` 只在 `check.py` 中读取用于比对，编码器本身不接触该文件；
实现中没有任何针对固定输入的缓存或特判，任意批大小与常见图像格式均可运行。

## 实验报告

`report/实验报告-项目1-NumPy图像编码器.docx`（深圳大学实验报告模板的封面与批阅页，正文 5 页）与同名 PDF
（Word 导出，共 7 页：封面、正文 5 页、批阅页），两份内容一致。

正文按作业要求组织为六节：1 任务理解与实验准备、2 从像素到图片块特征、3 Transformer 块与最终输出、
4 结果验证与问题分析、5 学习反思与后续改进、6 本人参与与辅助说明；证据为表 2（分阶段误差）、
表 3（鲁棒性测试）、图 1（误差与容差对比）、图 2（输出余弦相似度）。
