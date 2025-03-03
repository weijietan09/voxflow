# 贡献指南

欢迎参与 VoxFlow！这是一个研究早期的项目，接口和结构都还在变动，
小而聚焦的改动最容易合入。

## 开发环境

```bash
git clone https://github.com/weijietan09/voxflow.git
cd voxflow
pip install -e ".[dev]"
pip install torch --index-url https://download.pytorch.org/whl/cpu
pre-commit install
```

## 提交前检查

```bash
ruff check . && ruff format .
mypy src/voxflow
pytest --cov=voxflow
```

CI 会在 Python 3.10 / 3.11 / 3.12 上跑同样的检查，请确保本地全绿。

## 约定

- 提交粒度尽量小，一次只做一件事；提交信息说明「做了什么、为什么」。
- 新功能请配套测试；改 bug 请附一个能复现的回归测试。
- 公开函数写类型标注与简要 docstring；注释解释「为什么」，而非「是什么」。
- 面向用户的文档 / 注释用中文，与仓库整体保持一致。

## 提交 PR

1. 从 `main` 切分支。
2. 完成改动并补测试 / 文档。
3. 本地跑通上面的检查。
4. 发起 PR，说明动机与影响，关联相关 issue。

## 报告问题

用 issue 模板描述复现步骤、参考音频 / 文本、期望与实际行为、环境信息。
安全相关问题请勿公开，见 [SECURITY.md](./SECURITY.md)。
