# 🧬 2025 CADD课程实践平台

现代化计算机辅助药物设计工具套件，集成多种机器学习模型、高级分析功能和交互式可视化。

## ✨ 主要特性

### 🤖 多模型支持
- **随机森林 (Random Forest)**: 鲁棒性强，易于解释
- **支持向量机 (SVM)**: 适用于高维数据
- **XGBoost**: 高性能梯度提升
- **LightGBM**: 快速轻量级梯度提升
- **神经网络 (Neural Network)**: 深度学习方法

### 📊 现代化UI/UX
- 深蓝色+白色+绿色主题设计
- Plotly交互式图表
- 响应式布局设计
- 进度条和加载动画
- 图标化导航菜单

### 🔬 核心功能模块

#### 1. 数据展示与分析
- 多维度数据可视化
- 相关性热力图分析
- PCA降维可视化
- 分子描述符计算
- Lipinski五规则检查
- 缺失值分析

#### 2. 模型训练与比较
- 多模型并行训练
- 自动超参数调优
- 交叉验证评估
- ROC曲线比较
- 性能指标对比
- 混淆矩阵可视化

#### 3. 活性预测
- 单分子SMILES预测
- 分子编辑器集成
- 批量预测功能
- 预测置信度评估
- SHAP可解释性分析
- 结果导出功能

#### 4. 项目管理
- 项目版本控制
- 模型结果保存
- 性能报告生成
- 项目导出/导入

#### 5. 高级分析
- t-SNE/UMAP化学空间可视化
- 分子骨架分析
- 描述符多维分析
- 化学多样性评估

#### 6. 知识获取
- PubMed文献搜索
- AI智能文献分析
- CADD知识库
- 毒性预测指南

## 🚀 安装指南

### 环境要求
- Python 3.8+
- 建议使用Conda环境管理

### 1. 克隆仓库
```bash
git clone https://github.com/your-repo/cadd-streamlit.git
cd cadd-streamlit
```

### 2. 创建虚拟环境
```bash
conda create -n cadd python=3.9
conda activate cadd
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 安装RDKit (化学信息学库)
```bash
conda install -c conda-forge rdkit
```

### 5. 启动应用
```bash
streamlit run app.py
```

应用将在 `http://localhost:8501` 启动

## 📁 项目结构

```
cadd-streamlit/
├── app.py                 # 主应用文件
├── requirements.txt       # 依赖包列表
├── README.md             # 项目说明
├── data/                 # 示例数据目录
├── projects/             # 训练项目保存目录
└── docs/                 # 文档目录
```

## 🎯 使用指南

### 1. 准备数据
- 准备CSV格式的分子数据
- 确保包含SMILES列和活性标签列
- 数据应进行质量检查和预处理

### 2. 数据分析
1. 在"数据展示"页面上传数据
2. 查看数据概况和质量报告
3. 分析特征分布和相关性
4. 计算分子描述符

### 3. 模型训练
1. 在"模型训练"页面选择数据
2. 配置特征工程参数
3. 选择要训练的模型
4. 设置训练参数
5. 开始训练并比较性能

### 4. 活性预测
1. 选择已训练的项目
2. 输入单个SMILES或上传批量数据
3. 进行预测并查看结果
4. 分析预测可解释性

### 5. 项目管理
1. 查看所有训练项目
2. 下载模型和报告
3. 管理项目文件

## 🔧 配置选项

### 分子指纹类型
- **Morgan指纹**: 适用于结构相似性分析
- **MACCS键**: 适用于药物发现

### 模型参数
- 支持网格搜索超参数优化
- 交叉验证折数可配置
- 测试集比例可调节

### 可视化选项
- 多种颜色主题
- 交互式图表
- 高分辨率图像导出

## 📚 数据格式要求

### 训练数据格式
```csv
SMILES,Activity,Compound_ID
CCO,1,Compound_001
CC(C)O,0,Compound_002
...
```

### 预测数据格式
```csv
SMILES,Compound_ID
CCO,Test_001
CC(C)O,Test_002
...
```

## 🧮 支持的分子描述符

- **MolWt**: 分子量
- **LogP**: 脂水分配系数
- **NumHDonors**: 氢键供体数量
- **NumHAcceptors**: 氢键受体数量
- **TPSA**: 拓扑极性表面积
- **NumRotatableBonds**: 可旋转键数量
- **NumAromaticRings**: 芳香环数量
- **FractionCsp3**: SP3碳原子分数
- **HeavyAtomCount**: 重原子数量
- **RingCount**: 环数量

## 🔬 算法支持

### 机器学习算法
- 随机森林 (Random Forest)
- 支持向量机 (SVM)
- XGBoost
- LightGBM
- 多层感知机 (MLP)

### 降维算法
- 主成分分析 (PCA)
- t-SNE
- UMAP

### 评估指标
- 准确率 (Accuracy)
- AUC-ROC
- 精确率 (Precision)
- 召回率 (Recall)
- F1得分

## 🐛 故障排除

### 常见问题

1. **RDKit安装失败**
   ```bash
   conda install -c conda-forge rdkit
   ```

2. **Streamlit启动错误**
   ```bash
   pip install --upgrade streamlit
   ```

3. **内存不足**
   - 减少数据集大小
   - 调整批处理大小
   - 关闭其他应用程序

4. **模型训练慢**
   - 使用更少的模型
   - 减少交叉验证折数
   - 启用并行处理

## 📈 性能优化

### 数据处理优化
- 使用缓存装饰器 `@st.cache_data`
- 批量处理大数据集
- 并行计算分子描述符

### 模型训练优化
- 启用多线程训练
- 使用GPU加速 (如果可用)
- 早停机制防止过拟合

## 🤝 贡献指南

1. Fork项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 📄 许可证

本项目采用MIT许可证 - 详情请见 [LICENSE](LICENSE) 文件

## 👥 开发团队

- **项目负责人**: TJCADD团队
- **开发者**: 计算机辅助药物设计课程组
- **联系方式**: cadd@example.com

## 🆕 更新日志

### v2.0.0 (2025-01-XX)
- ✨ 全新UI设计和用户体验
- 🤖 多模型支持和比较
- 📊 交互式可视化图表
- 🔬 批量预测功能
- 🧠 SHAP可解释性分析
- 📚 知识库和文献分析
- 🚀 性能优化和缓存机制

### v1.0.0 (2024-XX-XX)
- 🎉 初始版本发布
- 基础的随机森林模型
- 简单的数据可视化
- SMILES分子预测

## 🔗 相关链接

- [RDKit官方文档](https://www.rdkit.org/)
- [Streamlit官方文档](https://docs.streamlit.io/)
- [机器学习药物发现教程](https://example.com)
- [CADD最佳实践指南](https://example.com)

## 💡 技术支持

如果您在使用过程中遇到问题，请：

1. 查看FAQ部分
2. 搜索已有的Issues
3. 创建新的Issue描述问题
4. 联系开发团队

---

**⭐ 如果这个项目对您有帮助，请给我们一个Star！**