# 6 服务器发布仓库核查（2026-10-05）

- 精简仓库：`/home/zzx23457/hhw/leju_rl`；训练工作区仍为独立的 `/home/zzx23457/hhw/LejuLab-Train`。前者跟踪约 520 个文件，包括 Lab 任务源码、S52/S53 本体与动作参考、训练/评测/导出脚本、配置和各轮奖励记录。
- 跟踪的权重仅 `checkpoints/kuavo_s52_stairs/model_92099.pt`（S53 来源 teacher 的发布副本）及 `checkpoints/kuavo_s53_stairs/model_92150.pt`（S53 已保留基线）；失败轮次 v67/v68 的权重**没有**加入发布仓库。具体本体和 SHA-256 以 [S52 状态页](S52_STAIRS_STATUS_20261004_ZH.md) 为准，不能凭同名 checkpoint 推断相同模型。
- v67/v68 最新结果、源码快照、完整奖励曲线 CSV/SVG、路线验收 JSON 和失败报告已通过 GitHub 连接发布到 [S52 训练记录](../training_records/s52/)。本地与服务器分析 README 已同步，SHA-256 一致。
- **内容已同步，历史仍分叉**：服务器发布仓库 `main` 保留 3 个本地提交：`421e418`（v38-v41）、`e551e9e`（v53-v54）、`dc9ec3c`（v55-v56）。这八轮的 44 个归档文件已通过 GitHub 连接追加到公开 `main`（提交 `6188d3e`），并逐文件与服务器 Git blob SHA 核对，44/44 一致；公开分支保留了后来新增的 v67/v68 内容。原 3 个服务器提交的**提交 SHA/拓扑**未直接推送，服务器 `main` 与公开 `main` 仍分叉。服务器和本机对 GitHub 的 SSH 均返回 `Permission denied (publickey)`；服务器 HTTPS fetch 未在本次连接窗口内完成。此次没有 force-push、reset 或覆盖任何分支。
- 发布仓库仍跟踪少量构建生成的 `source/leju_robot/lejuRobot.egg-info/*` 和部分历史任务资产；因此“只有必要文件”的进一步清理尚未彻底完成。清理前须核实干净克隆的任务注册、训练脚本与部署依赖，不能为了缩小仓库破坏可复现性。
- 未启动/停止训练、Docker、ROS 或真机相关进程；没有删除任何受保护权重。

待完成的是让服务器本地 `main` 非破坏性合并公开 `main`，并清理已跟踪的生成文件；历史归档内容已公开，不应重复上传。禁止覆盖已发布的 v67/v68 记录。
