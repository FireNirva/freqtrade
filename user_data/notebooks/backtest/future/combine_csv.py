import os
import pandas as pd

# 定义时间周期和趋势
periods = ["1h", "4h", "5m", "15m"]
trends = ["uptrend", "sideways", "downtrend"]

# 定义文件和输出目录
base_dir = "./result_overview"
OUTPUT_DIR = "result_overview"
OUTPUT_FILE_PREFIX = "result_overview"

# 创建输出目录如果不存在
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 合并每个趋势的数据
for trend in trends:
    combined_df = pd.DataFrame()
    
    # 读取每个时间周期的文件并添加时间周期
    for period in periods:
        file = os.path.join(base_dir, f"result_overview_{trend}_{period}.csv")
        if os.path.exists(file):
            df = pd.read_csv(file)
            df.iloc[:, 0] = df.iloc[:, 0] + f"_{period}"
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        else:
            print(f"文件 {file} 不存在，跳过。")
    
    # 保存合并后的数据框到新的CSV文件
    output_file_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILE_PREFIX}_{trend}.csv")
    combined_df.to_csv(output_file_path, index=False)
    print(f"文件合并完成，结果已保存到 {output_file_path}")
