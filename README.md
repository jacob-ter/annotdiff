# annotdiff

**比对两批标注结果，定位分歧样本。**

`annotdiff` —— 按样本比对两批标注，列出分歧点、一致率与分歧最集中的类别。

- 零依赖，Python 3.10+ 标准库即可运行
- 单文件，`annotdiff.py` 拷走就能用
- 不联网、不写文件

<!-- ipos:install -->
## 安装

零依赖，Python 3.10+ 即可：

```bash
pip install jacob-ter-annotdiff
```

装完直接用命令名调用：

```bash
annotdiff --help
```

也可以不用安装，直接把 `annotdiff.py` 拷走用——它只依赖标准库。

## 为什么需要它

标注工具多只报一致率；annotdiff 列出具体分歧样本与集中出现的类别，指出该补哪类标注规则。

它遵守下面几条自己定的规矩：

1. **一致率不告诉你在哪分歧**。这个工具列出具体分歧样本，以及分歧最集中的类别。
2. **分歧指向规则，不是指向人**。输出的用途是「该改哪条规则」。
3. **不同一致率算法都算一遍**。算法选不同，结论会差很多。

## 用法

```bash
python annotdiff.py --a a.jsonl --b b.jsonl
python annotdiff.py --a a.jsonl --b b.jsonl --field label --json
```

加 `--json` 可以拿到机器可读的结果，便于接到流水线里。

## 已知局限

- 一致率用简单的比例统计，不算 Cohen's kappa 之类的系数。
- 只比对两份标注，不做标注质量评估。
- 分歧只列出来，不替你裁决谁对。

## 跑测试

```bash
python test_annotdiff.py
```

测试用独立的断言与临时数据，不污染工作目录。

## 来源与参考

- Python 标准库文档：<https://docs.python.org/3/>
- GitHub 仓库：<https://github.com/jacob-ter/annotdiff>

---

## 关于作者

jacob-ter 做标注一致性的分歧定位相关的问题。这个工具先替你回答一个具体问题，再谈别的。

- 个人主页：<https://jacob-ter.github.io>
- 仓库：<https://github.com/jacob-ter/annotdiff>

## 许可

MIT，见 [LICENSE](LICENSE)。
