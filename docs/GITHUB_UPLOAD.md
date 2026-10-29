# 上传到 GitHub

以下命令在**仓库目录**执行。目标仓库是已创建的空仓库：`https://github.com/Leewin239/tilelang-distributed-layout`。本项目根目录没有 README。

## 1. 初始化、检查并提交

```bash
git init
git branch -M main
git add .gitignore kernels benchmarks docs third_party
git status --short
git commit -m "Organize distributed TileLang layout experiments"
```

执行 `git commit` 前，先检查 `git status --short`：只应出现本仓库的代码、说明和上游许可证，不应出现私钥、口令、服务器日志、数据集或虚拟环境。`.gitignore` 已排除常见的大文件和敏感文件，但仍以待提交文件清单为准。如需检查实际暂存内容，运行 `git diff --cached --stat` 和 `git diff --cached`。

如果 Git 提示没有姓名或邮箱，只为本仓库设置：

```bash
git config user.name "你的名字"
git config user.email "你的 GitHub 邮箱"
git commit -m "Organize distributed TileLang layout experiments"
```

## 2. 推送到已创建的空仓库

在本地执行：

```bash
git remote add origin https://github.com/Leewin239/tilelang-distributed-layout.git
git push -u origin main
```

GitHub HTTPS 推送通常通过浏览器/凭据管理器登录；不要把密码或访问令牌写进仓库文件或命令历史。如果以后决定公开仓库，请先确认新增代码的许可证，并保留 `third_party/TILESCALE_LICENSE.txt` 中的上游署名。

## 3. 后续更新

```bash
git add kernels benchmarks docs
git commit -m "Describe this experiment update"
git push
```

若需要从服务器更新实验结果，优先把精简后的结果写到 `docs/RESULTS.md`；不要直接把整个 `logs/`、虚拟环境或大型数据文件推送到 GitHub。
