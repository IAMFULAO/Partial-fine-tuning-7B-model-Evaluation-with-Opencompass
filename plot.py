import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def read_summary_file(file_path):
    """读取总结文件并提取数据"""
    try:
        # 读取Markdown表格
        df = pd.read_csv(file_path, sep='|', header=0)
        
        # 移除首尾的空列，选择第一列和最后一列
        df = df.iloc[:, [1, -2]]  # 第一列是dataset，最后一列是准确率
        
        # 清理列名
        df.columns = ['dataset', 'accuracy']
        
        # 清理数据
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        
        # 转换准确率为数值
        df['accuracy'] = pd.to_numeric(df['accuracy'], errors='coerce')

        # 检查并去除重复
        duplicates = df.duplicated('dataset', keep=False)
        if duplicates.any():
            duplicate_count = duplicates.sum()
            print(f"发现 {duplicate_count/2} 个重复数据集")
        
        df_dedup = df.drop_duplicates(subset='dataset', keep='first')
        
        return df_dedup.dropna()
        
    except Exception as e:
        print(f"读取文件错误: {e}")
        return None

def create_comparison_chart(before_file, after_file, output_dir=None):
    """创建对比图表"""
    
    # 确保输出目录存在
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"输出目录已创建: {output_dir}")

    # 读取文件
    df_before = read_summary_file(before_file)
    df_after = read_summary_file(after_file)
    
    if df_before is None or df_after is None:
        return None
    
    print(f"微调前: {len(df_before)} 个数据集")
    print(f"微调后: {len(df_after)} 个数据集")
    
    # 直接合并，不预先重命名
    merged_df = pd.merge(
        df_before, 
        df_after, 
        on='dataset',
        suffixes=('_before', '_after')
    )
    
    print(f"匹配数量: {len(merged_df)}")
    
    # 重命名列
    merged_df = merged_df.rename(columns={
        'accuracy_before': 'before_tuning',
        'accuracy_after': 'after_tuning'
    })
    
    # 计算提升
    merged_df['improvement'] = merged_df['after_tuning'] - merged_df['before_tuning']
    
    # 简化数据集名称
    merged_df['dataset_short'] = merged_df['dataset'].str.replace('lukaemon_mmlu_', '').str.replace('cmmlu-', '')
    
    # 分类：区分CMMLU和MMLU数据集
    merged_df['dataset_type'] = merged_df['dataset'].apply(
        lambda x: 'CMMLU' if x.startswith('cmmlu') else 'MMLU'
    )
    
    # 分别处理CMMLU和MMLU数据
    cmmlu_df = merged_df[merged_df['dataset_type'] == 'CMMLU'].copy()
    mmlu_df = merged_df[merged_df['dataset_type'] == 'MMLU'].copy()
    
    # 分别按提升幅度排序
    cmmlu_df = cmmlu_df.sort_values('improvement', ascending=False)
    mmlu_df = mmlu_df.sort_values('improvement', ascending=False)
    
    print(f"CMMLU数据集数量: {len(cmmlu_df)}")
    print(f"MMLU数据集数量: {len(mmlu_df)}")
    
    # 创建汇总图片（包含所有数据集）
    create_summary_plot(merged_df, output_dir)
    
    # 创建分类图片
    create_category_plots(cmmlu_df, mmlu_df, output_dir)
    
    # 打印统计信息
    print("\n" + "="*50)
    print("性能提升统计摘要")
    print("="*50)
    print(f"平均准确率 - 微调前: {merged_df['before_tuning'].mean():.2f}%")
    print(f"平均准确率 - 微调后: {merged_df['after_tuning'].mean():.2f}%")
    print(f"平均提升幅度: {merged_df['improvement'].mean():.2f}%")
    print(f"最大提升: {merged_df['improvement'].max():.2f}%")
    print(f"最小提升: {merged_df['improvement'].min():.2f}%")
    print(f"提升数据集数量: {(merged_df['improvement'] > 0).sum()}/{len(merged_df)}")
    print(f"下降数据集数量: {(merged_df['improvement'] < 0).sum()}/{len(merged_df)}")
    
    print(f"\nCMMLU平均提升: {cmmlu_df['improvement'].mean():.2f}%")
    print(f"MMLU平均提升: {mmlu_df['improvement'].mean():.2f}%")
    
    if output_dir:
        output_path = os.path.join(output_dir, 'summary.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*50 + "\n")
            f.write("性能提升统计摘要\n")
            f.write("="*50 + "\n")
            f.write(f"平均准确率 - 微调前: {merged_df['before_tuning'].mean():.2f}%\n")
            f.write(f"平均准确率 - 微调后: {merged_df['after_tuning'].mean():.2f}%\n")
            f.write(f"平均提升幅度: {merged_df['improvement'].mean():.2f}%\n")
            f.write(f"最大提升: {merged_df['improvement'].max():.2f}%\n")
            f.write(f"最小提升: {merged_df['improvement'].min():.2f}%\n")
            f.write(f"提升数据集数量: {(merged_df['improvement'] > 0).sum()}/{len(merged_df)}\n")
            f.write(f"下降数据集数量: {(merged_df['improvement'] < 0).sum()}/{len(merged_df)}\n")
            f.write(f"\nCMMLU平均提升: {cmmlu_df['improvement'].mean():.2f}%\n")
            f.write(f"MMLU平均提升: {mmlu_df['improvement'].mean():.2f}%\n")
            
            f.write("\n详细数据预览:\n")
            sorted_df = merged_df.sort_values('improvement', ascending=False)
            sorted_df = sorted_df.reset_index(drop=True)
            preview_df = sorted_df[['dataset_short', 'before_tuning', 'after_tuning', 'improvement', 'dataset_type']].head(10)
            f.write(preview_df.to_string(index=False) + "\n")

    return merged_df

def add_improvement_labels(ax, x_positions, before_values, after_values, improvement_values):
    """添加提升幅度标签"""
    for i, (x, before, after, improvement) in enumerate(zip(x_positions, before_values, after_values, improvement_values)):
        y_pos = max(before, after) + 1 
        ax.text(x, y_pos, f'{improvement:+.1f}%', 
                ha='center', va='bottom', fontsize=8, color='gray', rotation=0)

def create_summary_plot(merged_df, output_dir):
    """创建汇总图片"""
    # 按提升幅度排序
    merged_df_sorted = merged_df.sort_values('improvement', ascending=False)
    
    # 创建汇总图
    plt.figure(figsize=(20, 10))
    ax = plt.gca()
    
    # 设置位置和宽度
    x = np.arange(len(merged_df_sorted))
    width = 0.35
    
    # 创建条形图
    bars_before = plt.bar(x - width/2, merged_df_sorted['before_tuning'], width, 
                         label='before', color='#1f77b4', alpha=0.8, edgecolor='black')
    bars_after = plt.bar(x + width/2, merged_df_sorted['after_tuning'], width, 
                        label='after', color='#ff7f0e', alpha=0.8, edgecolor='black')
    
    # 设置图表属性
    plt.xlabel('Datasets', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Model performance before and after fine-tuning (Summary)', fontsize=16, fontweight='bold', pad=20)
    
    # 设置x轴标签
    plt.xticks(x, merged_df_sorted['dataset_short'], rotation=45, ha='right', fontsize=8)
    
    # 设置y轴范围
    y_max = max(merged_df_sorted[['before_tuning', 'after_tuning']].max().max(), 60)
    plt.ylim(0, y_max + 5)  
    
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # 保存汇总图
    if output_dir:
        output_path = os.path.join(output_dir, 'summary_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"汇总图已保存至: {output_path}")
    
    plt.show()

def create_category_plots(cmmlu_df, mmlu_df, output_dir):
    """创建分类图片"""
    
    # 创建CMMLU图表
    if len(cmmlu_df) > 0:
        create_single_category_plot(cmmlu_df, 'CMMLU', output_dir)
    
    # 创建MMLU图表
    if len(mmlu_df) > 0:
        create_single_category_plot(mmlu_df, 'MMLU', output_dir)

def create_single_category_plot(df, category_name, output_dir, max_per_plot=30):
    """创建单个分类的图表"""
    n_datasets = len(df)
    
    # 如果数据集数量超过最大值，分成多个子图
    if n_datasets <= max_per_plot:
        # 单个图
        plt.figure(figsize=(16, 10))
        ax = plt.gca()
        
        x = np.arange(n_datasets)
        width = 0.35
        
        bars_before = plt.bar(x - width/2, df['before_tuning'], width, 
                             label='before', color='#1f77b4', alpha=0.8, edgecolor='black')
        bars_after = plt.bar(x + width/2, df['after_tuning'], width, 
                            label='after', color='#ff7f0e', alpha=0.8, edgecolor='black')
        
        add_improvement_labels(ax, x, 
                              df['before_tuning'].values,
                              df['after_tuning'].values,
                              df['improvement'].values)
        
        plt.xlabel('Datasets', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title(f'{category_name} Datasets - Model performance before and after fine-tuning', fontsize=14, fontweight='bold')
        
        plt.xticks(x, df['dataset_short'], rotation=45, ha='right', fontsize=8)
        
        y_max = max(df[['before_tuning', 'after_tuning']].max().max(), 60)
        plt.ylim(0, y_max + 5)  
        
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        # 保存图片
        if output_dir:
            output_path = os.path.join(output_dir, f'{category_name.lower()}_comparison.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"{category_name}图已保存至: {output_path}")
        
        plt.show()
        
    else:
        # 多个子图
        n_plots = (n_datasets + max_per_plot - 1) // max_per_plot
        
        for plot_idx in range(n_plots):
            start_idx = plot_idx * max_per_plot
            end_idx = min((plot_idx + 1) * max_per_plot, n_datasets)
            plot_df = df.iloc[start_idx:end_idx]
            
            plt.figure(figsize=(16, 10))
            ax = plt.gca()
            
            x = np.arange(len(plot_df))
            width = 0.35
            
            bars_before = plt.bar(x - width/2, plot_df['before_tuning'], width, 
                                 label='before', color='#1f77b4', alpha=0.8, edgecolor='black')
            bars_after = plt.bar(x + width/2, plot_df['after_tuning'], width, 
                                label='after', color='#ff7f0e', alpha=0.8, edgecolor='black')
            
            # 添加提升幅度标签
            add_improvement_labels(ax, x, 
                                  plot_df['before_tuning'].values,
                                  plot_df['after_tuning'].values,
                                  plot_df['improvement'].values)
            
            plt.xlabel('Datasets', fontsize=12)
            plt.ylabel('Accuracy (%)', fontsize=12)
            plt.title(f'{category_name} Datasets - Model performance before and after fine-tuning (Part {plot_idx+1}/{n_plots})', 
                     fontsize=14, fontweight='bold')
            
            plt.xticks(x, plot_df['dataset_short'], rotation=45, ha='right', fontsize=8)
            
            y_max = max(plot_df[['before_tuning', 'after_tuning']].max().max(), 60)
            plt.ylim(0, y_max + 5) 
            
            plt.legend()
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            # 保存图片
            if output_dir:
                output_path = os.path.join(output_dir, f'{category_name.lower()}_comparison_part{plot_idx+1}.png')
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"{category_name}图第{plot_idx+1}部分已保存至: {output_path}")
            
            plt.show()

if __name__ == "__main__":
    # 文件路径
    before_file = "/home/liruizheng/Partial_fine_tuning_task/opencompass/outputs/llama2_7b_chat_eval/20251213_223332/summary/summary_20251213_223332.md"
    after_file = "/home/liruizheng/Partial_fine_tuning_task/opencompass/outputs/llama2_7b_chat_lora_eval/20251215_233514/summary/summary_20251215_233514.md"
    
    # 输出目录
    output_dir = "/home/liruizheng/Partial_fine_tuning_task/comparison/origin-lora-0"
    
    # 创建对比图表
    result_df = create_comparison_chart(before_file, after_file, output_dir)
    
    if result_df is not None:
        sorted_df = result_df.sort_values('improvement', ascending=False)
        sorted_df = sorted_df.reset_index(drop=True)
        print("\n详细数据预览:")
        print(sorted_df[['dataset_short', 'before_tuning', 'after_tuning', 'improvement', 'dataset_type']].head(10))