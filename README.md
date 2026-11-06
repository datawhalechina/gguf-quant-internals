<h1 align="center">GGUF 量化内幕</h1>

> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

本教程涵盖常用热门量化格式 Q2~8K、IQ、UD_IQ，以及最新的三值量化 PTQ1和 STQ1。带有 C++/C#/Python 三种代码解释概念，配有 ASCII Art 图表。

## 项目意义

GGUF 量化技术之所以成为端侧大模型部署和大模型技术普及的关键基础设施，在于它把原本动辄数十GB、依赖高带宽显存和云端算力的模型权重，压缩到手机、笔记本、嵌入式设备可承受的存储与内存带宽范围内。

通过 Q4_0、Q4_K、Q6_K、IQ 码本以及 STQ1/PTQ1 等三值格式的分层设计，GGUF 能在 1.5–6.5 bit/权重之间按张量敏感度灵活取舍，用块内尺度、超级块两级尺度、重要性矩阵和查表码本在显著减小体积的同时尽量保住模型质量，并借助 mmap 加载、逐张量混合精度和 llama.cpp 工具链实现“转换—量化—本地推理”的低门槛闭环。

没有这类量化，端侧部署将受制于内存、功耗和成本而难以落地，大模型也只能停留在云端少数玩家的手中，正是 GGUF 量化让离线、低延迟、隐私可控的本地推理成为现实，从而推动大模型从集中式服务走向更广泛的技术普及。

## 项目受众

想要制作端侧推理引擎，或者制作 GGUF 量化模型的人。

## 在线阅读
https://datawhalechina.github.io/gguf-quant-internals

## 目录

|  章节名   | 简介 | 状态 |
|  ----  | ---- | ---- |
| [第1章 xxx](https://github.com/datawhalechina/repo-template/blob/main/docs/chapter1)  | xxx | ✅ |
| [第2章 xxx](https://github.com/datawhalechina/repo-template/blob/main/docs/chapter2)  | xxx | ✅ |
| [第3章 xxx](https://github.com/datawhalechina/repo-template/blob/main/docs/chapter3)  | xxx | ✅ |
| 第4章  | xxx | 🚧 |

## 贡献者名单

| 姓名 | 职责 | 简介 |
| :----| :---- | :---- |
| 张子辰 | 项目负责人 | 专注于模型量化、端侧推理和注意力机制改进 |

## 参与贡献

- 如果你发现了一些问题，可以提Issue进行反馈，如果提完没有人回复你可以联系[保姆团队](https://github.com/datawhalechina/DOPMC/blob/main/OP.md)的同学进行反馈跟进~
- 如果你想参与贡献本项目，可以提Pull Request，如果提完没有人回复你可以联系[保姆团队](https://github.com/datawhalechina/DOPMC/blob/main/OP.md)的同学进行反馈跟进~
- 如果你对 Datawhale 很感兴趣并想要发起一个新的项目，请按照[Datawhale开源项目指南](https://github.com/datawhalechina/DOPMC/blob/main/GUIDE.md)进行操作即可~

## 关注我们

<div align=center>
<p>扫描下方二维码关注公众号：Datawhale</p>
<img src="https://raw.githubusercontent.com/datawhalechina/pumpkin-book/master/res/qrcode.jpeg" width = "180" height = "180">
</div>

## LICENSE

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a><br />本作品采用<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议</a>进行许可。
