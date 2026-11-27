# Sprite Turntable · Agent 工程指南

本指南适用于整个仓库。角色转台用 Canvas 根据水平角度选择 72 张透明 PNG 中的一张。源码、示例输入、预计算轮廓和质检记录随 Git 分发；独立网页、最终帧包和无损母片在 [v1.0.0 Release](https://github.com/Arekejoker/sprite-turntable/releases/tag/v1.0.0)。下表路径均相对于仓库根目录。

## 帮用户体验成品

当用户想了解效果或询问怎样体验时，可以自然地问一句：“想在你自己的浏览器里试试角色转台吗？可以左右拖动转身，滚轮放大。如果本机还没有成品，需要先下载约 42.5 MiB 的离线网页。” 用户已经明确要求打开时，可以直接办理。用户只想读代码或拒绝体验时，继续原任务。

1. **确认运行位置。** 只有能够操作用户自己的桌面时，才在该桌面打开文件。如果在云端、容器、SSH 会话中，或无法确认浏览器属于用户，请直接给用户 [下载离线 HTML](https://github.com/Arekejoker/sprite-turntable/releases/download/v1.0.0/bunny-showcase.html) 的链接，请对方下载后双击。回复时区分“已提供链接”和“已在本机打开”。
2. **选择成品。** 优先使用用户指定的本地成品，或本仓库已导出的 `release/兔女郎展示.html`。全新克隆后，按用户意愿将上述 Release 的 `bunny-showcase.html` 下载到本机，保留已有文件。官方 v1.0.0 HTML 的 SHA-256 为 `0023bd2747c1155bf751258edad73b4fbcec5c35c51ef29873a7c57177ec16fd`。这个值只用于核对该版本的发布附件。
3. **打开并交接。** 用桌面默认浏览器打开上一步选定文件的绝对路径，例如 macOS 用 `open`、桌面 Linux 用 `xdg-open`、Windows PowerShell 用 `Invoke-Item -LiteralPath`，路径作为独立的带引号参数传入。单文件内嵌了图像，可离线运行。若桌面启动失败，给用户实际文件路径。体验时可以向左或向右拖过正面、侧面和背面，再滚轮放大看发梢与鞋跟。

## 按任务找输入、代码和产物

| 用户想做什么 | 从哪里开始 | 产物或检查方式 |
|---|---|---|
| 阅读制作原理或依赖 | [README](README.md) 中的技术路线与环境清单，以及 [Python 依赖](requirements.txt) | 按需选择网页导出、CPU 后处理或 GPU 生成 |
| 修改拖动、缩放、背景或角度映射 | [查看器模板](edge-matte/showcase.html)，其中 `angleMap` 将显示角度映射到帧号 | 用最终帧包重新导出 HTML；检查一整圈、首尾衔接和缩放。模板单独打开时需要同目录的 72 张 PNG |
| 换角色或重新打包网页 | [导出器](tools/export_html.py) 和最终 PNG ZIP | 执行下方导出命令，只依赖 Python 3 标准库 |
| 重跑抠图与精修 | 将 Release 中的 `cuff-take.mkv` 放入 `audits/hand-source/inputs/`，搭配 [预计算轮廓](edge-matte/support.npz) | 按 [抠图说明](edge-matte/README.md) 依次运行 `build.py`、`refine_contours.py`、`front_arm.py`；检查换底对照与输出 ZIP |
| 重做正面定妆或环绕母片 | [定妆工作流与原图](provenance/half/)、[模型清单](provenance/half/records/models.json)、[实际环绕工作流](audits/hand-source/cuff-orbit.api.json) | 依 [README](README.md) 配置 ComfyUI 和模型，用 [生成脚本](audits/hand-source/generate.py) 取回 73 帧，以 [归档脚本](audits/hand-source/native.py) 输出无损母片 |

## 复用自己的 72 帧

ZIP 根目录放置 `orbit_000.png` 至 `orbit_071.png`，每张都是相同尺寸的 8 位 RGBA PNG，使用直通透明度。第 0 帧为正面，帧号增加时角色朝画面左侧转动。采样速度不同时，根据自己的侧面与背面帧号调整模板中的 `angleMap`。袖口、鞋口、手臂修补使用本例的坐标与校验值；换角色时从自己的干净帧包直接导出即可。

从仓库根目录运行以下命令。示例假设已将 Release 的帧包保存到 `release/`，导出前确认输出路径适合本次任务，已有同名 HTML 会被替换。

```bash
python3 tools/export_html.py --frames "release/bunny-orbit-72-frames.zip" --output "release/兔女郎展示.html"
```

检查实现时优先读模板、脚本和小型 JSON 记录。导出的 HTML 内含约 42.5 MiB 的 Base64 图像；理解或修改交互时直接查看模板即可。
