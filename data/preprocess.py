import pandas as pd
from datasets import load_dataset
from pathlib import Path

# 以项目根目录为基准
ROOT = Path(__file__).parent.parent
data_dir = ROOT / "data"
data_dir.mkdir(exist_ok=True)

# 加载 Enron 数据集（英文）
dataset = load_dataset("SetFit/enron_spam", split='train')   # 可用子集

# 只保留正文和主题
df = pd.DataFrame(dataset)
df = df[['subject', 'message']]
df = df.dropna()
df = df[df['subject'].str.len() > 5]
df = df[df['message'].str.len() > 50]

# 保存文件
df.to_csv(data_dir / "enron_clean.csv", index=False)
