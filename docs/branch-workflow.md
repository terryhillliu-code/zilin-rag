# Branch 工作流方案

## 问题

Claude Code 的 `/branch` 创建的是**独立会话**，没有合并机制：
- 分支的文件修改和 git commit 可见（同文件系统）
- 对话上下文、任务状态、memory 更新**完全隔离**
- 无法将分支的工作带回主会话

## 解决方案：用 Git 替代 /branch

**原则：永远用 git branch + worktree，不用 /branch**

### 日常开发流程

```bash
# 1. 创建功能分支
cd ~/zhiwei-rag && git checkout -b feature/xxx

# 2. 在新会话中工作（指定分支）
claude -c --continue -p "当前在 feature/xxx 分支工作"

# 3. 完成后提交
git add -A && git commit -m "feat: xxx"

# 4. 切回主线，合并
git checkout main && git merge feature/xxx

# 5. 清理分支
git branch -d feature/xxx
```

### 并行任务（多任务同时开发）

```bash
# 任务 A: 用 worktree 隔离
cd ~/zhiwei-rag
git worktree add ../zhiwei-rag-task-a feature/task-a

# 新会话中指定 worktree
claude -w ../zhiwei-rag-task-a

# 完成后
git worktree remove ../zhiwei-rag-task-a
git branch -D feature/task-a
```

### 分支命名约定

| 前缀 | 用途 | 示例 |
|------|------|------|
| `feature/` | 新功能 | `feature/oss-integration` |
| `fix/` | Bug 修复 | `fix/quota-tracker` |
| `experiment/` | 探索性尝试 | `experiment/new-rag-pipeline` |
| `backup/` | 大改前备份 | `backup/pre-refactor-20260412` |

### 什么时候用 /branch

仅用于**决策树**场景：想尝试两种方案，不确定哪种更好，不想丢失当前上下文。

```
主会话: 方案 A → 失败了 → 回主会话试方案 B
分支:   方案 B
```

但这仍然不推荐，因为分支的上下文无法带回。更好的做法是：

```
主会话: 做方案 A
新会话: 做方案 B（独立启动，不 branch）
```

### Memory 注意事项

分支会话写入的 memory 文件（`~/.claude/projects/-Users-liufang/memory/`）会**持久化到磁盘**，所以主会话也能看到。但对话上下文和任务状态不会共享。

## 各仓库操作

```bash
# 知微系统涉及的所有仓库
for repo in zhiwei-rag zhiwei-bot zhiwei-scheduler zhiwei-docs arxiv-paper-analyzer; do
  cd ~/$repo
  git checkout -b feature/xxx
done
```
