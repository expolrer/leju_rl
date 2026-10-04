# S52 v68 FutureContactPhaseResidual 失败轮次记录

整理日期：2026-10-04。来源为 6 服务器真实训练归档和固定 seed=131 的无界面 Isaac/PhysX rollout，不是 MuJoCo 楼梯验收。

## 运行与奖励

- 任务：`Tracking-Stairs-FutureContactPhaseResidual-KuavoS52`；验收：同名 `-Play` 任务。
- 运行目录：`/home/zzx23457/hhw/LejuLab-Train/logs/rsl_rl/kuavoS52_stairs_future_contact_residual/2026-09-14_01-11-54_s52_future_contact_phase_residual_preflight128x8_v68_20260914`。
- 训练设置：128 环境，8 次更新，共 24,576 timesteps；最终 `model_92157.pt`。它是失败候选，不是公开发布的部署模型。
- TensorBoard mean reward：首点 6.60238，末点 13.4502；step 92153 最低 -7.628999，末点也是峰值；7 点平滑均值约 6.634。episode length 首点 22，末点 119.8；teacher action RMSE 约 2.2e-8 到 1.34e-7。
- 完整 TensorBoard 标量 CSV/JSON 与诊断 PNG 位于服务器归档 `analysis/s52_transfer/training_records/v68_future_contact_phase_residual_preflight128x8_20260914/curves`。最初分析命令缺少 `--rollout-dir` 已补跑。

## 真实物理验收

使用最终 `model_92157.pt`、seed=131、1351 步、无界面 Isaac/PhysX 回放。参考动作推进到最后一帧，失败重置为 0；足部高度覆盖 0/0.13/0.26/0.39/0.52 m 阶段。但滑移 p95 为 0.20255 m/s，同源最小距离仅 0.06486 mm，峰值足力约 2359.7 N，同源 clearance cost mean 为 0.009456。**不采用最终权重**。项目绝对路线分析器确认首轮终点误差 **0.34523 m**，超过 `<0.25 m` 验收门；终点双脚接触比例 1.0，完整 Lab 楼梯验收为 `false`。首轮 p95 滑移 0.21863 m/s、首轮峰值足力 1630.1 N；它们与前述全回放指标口径不同。零重置不能代替完整验收。

现有参考适配分析器的 root RMSE 被首帧世界坐标/局部坐标混用污染，不据此判定路线失败或成功。将物理验收和完整曲线放在一起看，末点 reward 升高并没有保证更大的最坏净空或更低冲击；需要在后续版本直接约束接触/净空风险，而非仅追求平均回报。本轮不生成或发布失败模型 MP4。

## 保留与下一步

源文件快照与本报告存入本目录；服务器归档保留实际曲线与 NPZ/JSON。受保护的 S52 v22 `model_92150.pt`（SHA-256 `93c457de92f9102125d7c88f8721e25a35e83caa03e4112d7a01074c78e50f56`）继续作为 warm-start，而非最终已通过楼梯模型。完整 TensorBoard 曲线 CSV、标量汇总 JSON、四图 SVG 和绝对路线验收 JSON 已发布在本目录；服务器归档仍保存原始图和 NPZ。
