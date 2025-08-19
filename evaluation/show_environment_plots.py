
"""
环境可视化图片汇总
"""
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def show_all_plots():
    """显示所有保存的图片"""
    plot_dir = 'save'
    
    if not os.path.exists(plot_dir):
        print("save 文件夹不存在！")
        return
    
    # 获取所有图片文件
    image_files = [f for f in os.listdir(plot_dir) if f.endswith('.png')]
    
    if not image_files:
        print("没有找到图片文件！")
        return
    
    print(f"找到 {len(image_files)} 张图片:")
    for i, filename in enumerate(sorted(image_files), 1):
        print(f"{i}. {filename}")
    
    # 创建图片展示
    n_images = len(image_files)
    cols = 3
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, filename in enumerate(image_files):
        row = i // cols
        col = i % cols
        
        img_path = os.path.join(plot_dir, filename)
        img = mpimg.imread(img_path)
        
        axes[row, col].imshow(img)
        axes[row, col].set_title(filename, fontsize=10)
        axes[row, col].axis('off')
    
    # 隐藏多余的子图
    for i in range(n_images, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('figures/environment_plots/plot_summary.png', bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == "__main__":
    show_all_plots()
