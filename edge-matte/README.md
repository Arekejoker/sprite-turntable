# 透明边缘抠图与精修说明

本目录包含将 MiniMax H3 生成的 73 帧中性灰底全身环绕视频拆分为 72 张 `640 × 1920` 直通透明度（straight alpha）RGBA PNG、并完成局部修边与网页交互模板的全部脚本。

## 文件说明

| 文件 | 作用 |
| --- | --- |
| [showcase.html](showcase.html) | 纯 HTML5 Canvas 360° 交互查看器模板（支持外置 PNG 与内嵌 Base64 单文件两种模式） |
| [track.py](track.py) | 使用 SAM 2.1 Hiera Tiny 从第 0 帧与第 72 帧双向传播正面遮罩，输出 `support.npz` |
| [matte.py](matte.py) | 基于每行灰底色差与邻近前景颜色求解边缘覆盖率，保护反光兔耳、发梢、指缝与细高跟 |
| [measure_instep.py](measure_instep.py) / [instep.py](instep.py) | 测量鞋沿边界并用原始定妆图袜面纹理补回正面右侧鞋口 |
| [build.py](build.py) | 串联 `matte.py`、`instep.py` 与 `verify.py`，输出初版 72 帧透明 PNG 包与换底检查页 |
| [refine_contours.py](refine_contours.py) | 在预乘透明度空间沿轮廓切向做抗锯齿滤波，平滑肩臂像素台阶 |
| [front_arm.py](front_arm.py) | 从未抠像的原始定妆图重测正面双臂外缘并渐变衔接邻近视角，输出最终 72 帧 PNG 包 |
| [verify.py](verify.py) | 逐帧核对透明 PNG、双底合成结果与基准指纹 |
| [support.npz](support.npz) | 预计算好的 72 帧双向追踪二值轮廓 |

## 复现顺序

先从 **Releases** 下载无损母片 `cuff-take.mkv` 放入 `audits/hand-source/inputs/`，然后在仓库根目录按顺序执行：

```bash
# 可选：若需用 GPU 重新跑 SAM 2.1 双向追踪
python3 edge-matte/track.py /path/to/sam2.1_hiera_tiny.pt

# 1. 生成初版 72 帧透明 PNG（内置 matte.py + instep.py + verify.py）
python3 edge-matte/build.py

# 2. 轮廓抗锯齿
python3 edge-matte/refine_contours.py

# 3. 正面双臂精修，产出最终 release/全身环绕_正面手臂精修_72帧.zip
python3 edge-matte/front_arm.py

# 4. 打包单文件离线 HTML
python3 tools/export_html.py
```

如需重新测算鞋沿坐标，可在运行 `build.py` 之前先执行 `python3 edge-matte/measure_instep.py`。
