# 贡献指南

改动前后请跑一遍仓库自检，它只检查**已跟踪的文件**里有没有客户数据。

## 数据保密

工位数模、由它派生的一切（网格、点云、视点坐标）以及私密设计笔记**绝不入库**。
`input/`、`output/`、`spike/`、`reference/00_Output Collection/` 全部被 gitignore；`docs/` 下的文档本身入库，只挡其中的私密子目录。并且有自检脚本会在这些东西**一旦被跟踪**时报错：

```bash
uv run python scripts/check_repo_hygiene.py
```

提交前请先跑。被忽略的文件之后仍然可以用 `git add -f` 加进来，但**已经推送出去的
收不回来**。
