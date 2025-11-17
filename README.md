# 📧 Smart Email Subject Generator  
**一句话生成专业邮件主题 —— 基于 T5-small 微调 + Flask 实时 Demo**

---

 ✨ 特性一览
- **Base Model**: [t5-small](https://huggingface.co/t5-small)（60 M 参数，MIT 许可）
- **数据**: 29 k 封 Enron 邮件清洗子集，**750 GB → 30 MB 微调**
- **训练**: 3 epoch，Beam Search 4，FP16 加速，**30 min 完成**
- **部署**: Flask 一键启动，**浏览器实时生成**
- **路径**: 纯相对设计，**中文目录可迁移英文环境**
- **依赖**: PyTorch + Hugging Face + Flask，**pip 一键安装**

---

## 🚀 一键运行（3 分钟）

```bash
# 1. 克隆（或下载 ZIP）
git clone https://github.com/YOUR_NAME/Email-Subject-Generator.git
cd Email-Subject-Generator

# 2. 激活虚拟环境
.\venv\Scripts\activate        # Windows
# source venv/bin/activate     # Linux/Mac

# 3. 启动网页
python app/app.py
```
浏览器打开 [http://127.0.0.1:5000](http://127.0.0.1:5000) → 粘贴邮件正文 → **一键生成主题**！

---

## 📸 浏览器 Demo
![demo](https://raw.githubusercontent.com/YOUR_NAME/Email-Subject-Generator/main/demo.png)

---

## 🧠 技术栈

| 模块 | 技术 | 备注 |
| ---- | ---- | ---- |
| **Base Model** | T5-small (60 M) | 官方 C4 语料预训练 |
| **微调数据** | Enron Email 子集 | 29 k 条，英文 |
| **训练框架** | PyTorch + Hugging Face | FP16 加速，3 epoch |
| **推理优化** | Beam Search 4，Early Stopping | 最大 20 tokens |
| **Web 框架** | Flask + Jinja2 | 相对路径设计 |
| **依赖管理** | requirements.txt | 一键安装 |

---

## 📁 项目结构（相对路径）

```
Email-Subject-Generator/
├── app/
│   ├── app.py                 # Flask 控制器
│   └── templates/
│       └── index.html         # 前端页面
├── model/
│   ├── train.py               # 微调脚本（3 epoch 示例）
│   └── inference.py           # 一行函数 generate_subject()
├── data/
│   └── enron_clean.csv        # 29 k 清洗邮件
├── saved_model/               # 训练产物（权重+tokenizer）
├── requirements.txt           # 依赖清单
├── demo.png                   # 浏览器截图
└── README.md                  # 本文件
```

---

## 🔧 一键训练（可选）

```bash
python model/train.py
# 产出：saved_model/*（权重+tokenizer）
```

---

## 📄 许可证
MIT © 2024 YOUR_NAME
