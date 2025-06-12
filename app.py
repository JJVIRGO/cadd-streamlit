import streamlit as st
from streamlit_ketcher import st_ketcher
import pandas as pd
import os
import re
import glob
import joblib
import random
import string
import io
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

# 机器学习和数据科学库
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import xgboost as xgb
import lightgbm as lgb
from tensorflow import keras
from tensorflow.keras import layers

# 可视化库
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# 化学信息学库
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFingerprintGenerator, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

# SHAP解释性AI
import shap

# 生物信息学和API
from Bio import Entrez
from openai import OpenAI

# PDF生成
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# 配置页面
st.set_page_config(
    page_title="2025 CADD课程实践平台",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    /* 主题配色 */
    :root {
        --primary-color: #1f4e79;
        --secondary-color: #2e8b57;
        --accent-color: #4CAF50;
        --background-color: #f8fafc;
        --text-color: #2c3e50;
    }
    
    /* 隐藏默认元素 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* 主容器样式 */
    .main-container {
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        text-align: center;
    }
    
    /* 卡片样式 */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 4px solid var(--primary-color);
    }
    
    /* 指标卡片 */
    .metric-card {
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem;
    }
    
    /* 按钮样式 */
    .stButton > button {
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* 侧边栏样式 */
    .css-1d391kg {
        background: linear-gradient(180deg, var(--primary-color), var(--secondary-color));
    }
    
    /* 进度条样式 */
    .stProgress > div > div > div > div {
        background: linear-gradient(45deg, var(--accent-color), var(--secondary-color));
    }
    
    /* 表格样式 */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# === 工具函数和配置 ===

class CADDConfig:
    """CADD平台配置类"""
    def __init__(self):
        self.models = {
            "随机森林": RandomForestClassifier,
            "支持向量机": SVC,
            "XGBoost": xgb.XGBClassifier,
            "LightGBM": lgb.LGBMClassifier,
            "神经网络": "neural_network"
        }
        
        self.model_params = {
            "随机森林": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 10, None],
                "max_features": ["sqrt", "log2"]
            },
            "支持向量机": {
                "C": [0.1, 1, 10],
                "kernel": ["rbf", "linear"],
                "gamma": ["scale", "auto"]
            },
            "XGBoost": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2]
            },
            "LightGBM": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2]
            }
        }
        
        # 动态检测可用的描述符
        self.descriptors = self._get_available_descriptors()
    
    def _get_available_descriptors(self):
        """动态检测可用的分子描述符"""
        base_descriptors = [
            "MolWt", "LogP", "NumHDonors", "NumHAcceptors", 
            "TPSA", "NumRotatableBonds", "NumAromaticRings",
            "HeavyAtomCount", "RingCount"
        ]
        
        # 检查 FractionCsp3 是否可用
        available_descriptors = base_descriptors.copy()
        
        # 测试分子用于检查描述符可用性
        test_mol = Chem.MolFromSmiles("CCO")  # 简单的乙醇分子
        
        if test_mol:
            # 检查 FractionCsp3 是否可用
            try:
                if hasattr(Descriptors, 'FractionCsp3'):
                    Descriptors.FractionCsp3(test_mol)
                    available_descriptors.append("FractionCsp3")
                elif hasattr(rdMolDescriptors, 'CalcFractionCsp3'):
                    rdMolDescriptors.CalcFractionCsp3(test_mol)
                    available_descriptors.append("FractionCsp3")
                else:
                    # 使用手动计算
                    available_descriptors.append("FractionCsp3")
            except:
                # FractionCsp3 不可用，跳过
                pass
        
        return available_descriptors

@st.cache_data
def load_config():
    """加载配置"""
    return CADDConfig()

# === 分子处理函数 ===

@st.cache_data
def calculate_molecular_descriptors(smiles: str) -> Dict[str, float]:
    """计算分子描述符"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    
    descriptors = {}
    
    # 定义描述符计算函数列表，包含名称和计算函数
    descriptor_functions = [
        ('MolWt', lambda m: Descriptors.MolWt(m)),
        ('LogP', lambda m: Descriptors.MolLogP(m)),
        ('NumHDonors', lambda m: Descriptors.NumHDonors(m)),
        ('NumHAcceptors', lambda m: Descriptors.NumHAcceptors(m)),
        ('TPSA', lambda m: Descriptors.TPSA(m)),
        ('NumRotatableBonds', lambda m: Descriptors.NumRotatableBonds(m)),
        ('NumAromaticRings', lambda m: Descriptors.NumAromaticRings(m)),
        ('HeavyAtomCount', lambda m: Descriptors.HeavyAtomCount(m)),
        ('RingCount', lambda m: Descriptors.RingCount(m)),
    ]
    
    # FractionCsp3 需要特殊处理（版本兼容性）
    def calculate_fraction_csp3(mol):
        """计算FractionCsp3，处理版本兼容性"""
        try:
            # 首先尝试从 Descriptors 模块
            if hasattr(Descriptors, 'FractionCsp3'):
                return Descriptors.FractionCsp3(mol)
            # 尝试从 rdMolDescriptors 模块
            elif hasattr(rdMolDescriptors, 'CalcFractionCsp3'):
                return rdMolDescriptors.CalcFractionCsp3(mol)
            # 手动计算 FractionCsp3
            else:
                return calculate_csp3_fraction_manual(mol)
        except:
            return None
    
    # 计算常规描述符
    for desc_name, desc_func in descriptor_functions:
        try:
            descriptors[desc_name] = desc_func(mol)
        except Exception as e:
            st.warning(f"计算描述符 {desc_name} 时出错: {e}")
            descriptors[desc_name] = None
    
    # 计算 FractionCsp3
    try:
        frac_csp3 = calculate_fraction_csp3(mol)
        if frac_csp3 is not None:
            descriptors['FractionCsp3'] = frac_csp3
        else:
            st.info("FractionCsp3 描述符暂不可用，已跳过")
    except Exception as e:
        st.warning(f"计算 FractionCsp3 时出错: {e}")
    
    # 移除 None 值
    descriptors = {k: v for k, v in descriptors.items() if v is not None}
    
    return descriptors

def calculate_csp3_fraction_manual(mol):
    """手动计算 FractionCsp3"""
    try:
        sp3_carbons = 0
        total_carbons = 0
        
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 6:  # 碳原子
                total_carbons += 1
                if atom.GetHybridization() == Chem.HybridizationType.SP3:
                    sp3_carbons += 1
        
        if total_carbons == 0:
            return 0.0
        return sp3_carbons / total_carbons
    except:
        return None

@st.cache_data
def mol_to_fingerprint(smiles: str, fp_type: str = "morgan") -> Optional[np.ndarray]:
    """生成分子指纹"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    try:
        if fp_type == "morgan":
            fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
            fp = fpgen.GetFingerprint(mol)
            arr = np.zeros((2048,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            return arr
        elif fp_type == "maccs":
            fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
            arr = np.zeros((167,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            return arr
    except Exception as e:
        st.error(f"生成指纹时出错: {e}")
        return None

@st.cache_data
def get_scaffold(smiles: str) -> Optional[str]:
    """获取分子骨架"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except:
        return None

# === 数据处理函数 ===

def clean_data_for_training(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """清理训练数据，处理NaN值和异常值"""
    
    # 数据清理信息
    cleaning_info = {
        'original_samples': len(X),
        'removed_samples': 0,
        'final_samples': 0,
        'nan_in_features': 0,
        'nan_in_targets': 0
    }
    
    # 检查输入数据
    if len(X) == 0 or len(y) == 0:
        st.error("数据为空，无法训练模型")
        return X, y, cleaning_info
    
    # 检查特征中的NaN
    nan_mask_X = np.isnan(X).any(axis=1)
    cleaning_info['nan_in_features'] = np.sum(nan_mask_X)
    
    # 检查目标变量中的NaN
    nan_mask_y = np.isnan(y)
    cleaning_info['nan_in_targets'] = np.sum(nan_mask_y)
    
    # 合并所有需要移除的样本
    invalid_mask = nan_mask_X | nan_mask_y
    
    if np.any(invalid_mask):
        st.warning(f"检测到 {np.sum(invalid_mask)} 个包含NaN值的样本，将被移除")
        
        # 移除包含NaN的样本
        X_clean = X[~invalid_mask]
        y_clean = y[~invalid_mask]
        
        cleaning_info['removed_samples'] = np.sum(invalid_mask)
        cleaning_info['final_samples'] = len(X_clean)
        
        # 检查清理后的数据
        if len(X_clean) == 0:
            st.error("清理后没有有效数据，请检查数据质量")
            return X, y, cleaning_info
        
        if len(X_clean) < 10:
            st.warning(f"清理后只有 {len(X_clean)} 个样本，可能影响模型性能")
        
        st.success(f"数据清理完成：保留 {len(X_clean)} 个有效样本（移除 {np.sum(invalid_mask)} 个无效样本）")
        
        return X_clean, y_clean, cleaning_info
    else:
        cleaning_info['final_samples'] = len(X)
        st.info("数据质量良好，无需清理")
        return X, y, cleaning_info

@st.cache_data
def preprocess_data(data: pd.DataFrame, feature_cols: List[str], 
                   target_col: str) -> Tuple[np.ndarray, np.ndarray]:
    """数据预处理"""
    # 移除缺失值
    clean_data = data.dropna(subset=feature_cols + [target_col])
    
    X = clean_data[feature_cols].values
    y = clean_data[target_col].values
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def validate_training_data(X: np.ndarray, y: np.ndarray) -> bool:
    """验证训练数据的有效性"""
    
    # 检查数据形状
    if X.shape[0] != y.shape[0]:
        st.error(f"特征矩阵和标签数量不匹配：X有{X.shape[0]}个样本，y有{y.shape[0]}个样本")
        return False
    
    # 检查最小样本数
    if len(X) < 10:
        st.error(f"样本数量过少（{len(X)}），需要至少10个样本进行训练")
        return False
    
    # 检查标签的类别
    unique_labels = np.unique(y)
    if len(unique_labels) < 2:
        st.error(f"标签类别不足，只有 {len(unique_labels)} 个类别，需要至少2个类别进行分类")
        return False
    
    # 检查类别平衡
    label_counts = pd.Series(y).value_counts()
    min_class_count = label_counts.min()
    if min_class_count < 2:
        st.warning(f"最少的类别只有 {min_class_count} 个样本，可能影响交叉验证")
    
    # 显示数据统计
    st.info(f"训练数据统计：{len(X)} 个样本，{X.shape[1]} 个特征")
    
    col1, col2 = st.columns(2)
    with col1:
        for i, (label, count) in enumerate(label_counts.items()):
            st.metric(f"类别 {label}", f"{count} 个样本")
    
    with col2:
        balance_ratio = min(label_counts) / max(label_counts)
        st.metric("类别平衡比", f"{balance_ratio:.3f}")
    
    return True

# === 模型训练函数 ===

def create_neural_network(input_dim: int, hidden_layers: List[int] = [128, 64, 32]) -> keras.Model:
    """创建神经网络模型"""
    model = keras.Sequential()
    model.add(layers.Dense(hidden_layers[0], activation='relu', input_dim=input_dim))
    model.add(layers.Dropout(0.3))
    
    for units in hidden_layers[1:]:
        model.add(layers.Dense(units, activation='relu'))
        model.add(layers.Dropout(0.3))
    
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

@st.cache_data
def train_multiple_models(X_train: np.ndarray, X_test: np.ndarray, 
                         y_train: np.ndarray, y_test: np.ndarray,
                         selected_models: List[str]) -> Dict[str, Any]:
    """训练多个模型并比较性能"""
    config = load_config()
    results = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, model_name in enumerate(selected_models):
        status_text.text(f"正在训练 {model_name}...")
        
        if model_name == "神经网络":
            model = create_neural_network(X_train.shape[1])
            history = model.fit(X_train, y_train, epochs=50, batch_size=32, 
                              validation_split=0.2, verbose=0)
            y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
            y_prob = model.predict(X_test).flatten()
        else:
            model_class = config.models[model_name]
            
            # 使用默认参数快速训练
            if model_name == "随机森林":
                model = model_class(n_estimators=100, random_state=42)
            elif model_name == "支持向量机":
                model = model_class(kernel='rbf', probability=True, random_state=42)
            elif model_name == "XGBoost":
                model = model_class(n_estimators=100, random_state=42)
            elif model_name == "LightGBM":
                model = model_class(n_estimators=100, random_state=42, verbose=-1)
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
        
        # 计算性能指标
        accuracy = accuracy_score(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = auc(fpr, tpr)
        
        # 交叉验证
        if model_name != "神经网络":
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
        else:
            cv_mean = cv_std = None
        
        results[model_name] = {
            'model': model,
            'accuracy': accuracy,
            'auc_score': auc_score,
            'fpr': fpr,
            'tpr': tpr,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'cv_mean': cv_mean,
            'cv_std': cv_std
        }
        
        progress_bar.progress((i + 1) / len(selected_models))
    
    status_text.text("模型训练完成！")
    return results

# === 可视化函数 ===

def create_model_comparison_plot(results: Dict[str, Any]) -> go.Figure:
    """创建模型比较图表"""
    models = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in models]
    auc_scores = [results[model]['auc_score'] for model in models]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('模型准确率比较', '模型AUC比较'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 准确率柱状图
    fig.add_trace(
        go.Bar(x=models, y=accuracies, name='准确率', 
               marker_color='lightblue', text=[f'{acc:.3f}' for acc in accuracies],
               textposition='auto'),
        row=1, col=1
    )
    
    # AUC柱状图
    fig.add_trace(
        go.Bar(x=models, y=auc_scores, name='AUC', 
               marker_color='lightgreen', text=[f'{auc:.3f}' for auc in auc_scores],
               textposition='auto'),
        row=1, col=2
    )
    
    fig.update_layout(
        title_text="模型性能比较",
        showlegend=False,
        height=400
    )
    
    return fig

def create_roc_comparison_plot(results: Dict[str, Any]) -> go.Figure:
    """创建ROC曲线比较图"""
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1
    
    for i, (model_name, result) in enumerate(results.items()):
        fig.add_trace(go.Scatter(
            x=result['fpr'], 
            y=result['tpr'],
            mode='lines',
            name=f"{model_name} (AUC = {result['auc_score']:.3f})",
            line=dict(color=colors[i % len(colors)], width=2)
        ))
    
    # 添加对角线
    fig.add_trace(go.Scatter(
        x=[0, 1], 
        y=[0, 1],
        mode='lines',
        name='随机分类器',
        line=dict(color='gray', dash='dash')
    ))
    
    fig.update_layout(
        title='ROC曲线比较',
        xaxis_title='假正率 (FPR)',
        yaxis_title='真正率 (TPR)',
        height=500,
        legend=dict(x=0.4, y=0.1)
    )
    
    return fig

def create_correlation_heatmap(data: pd.DataFrame) -> go.Figure:
    """创建相关性热力图"""
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    corr_matrix = data[numeric_cols].corr()
    
    fig = ff.create_annotated_heatmap(
        z=corr_matrix.values,
        x=list(corr_matrix.columns),
        y=list(corr_matrix.columns),
        annotation_text=corr_matrix.round(2).values,
        colorscale='RdBu',
        zmid=0
    )
    
    fig.update_layout(
        title='特征相关性热力图',
        height=600
    )
    
    return fig

def create_pca_plot(X: np.ndarray, y: np.ndarray) -> go.Figure:
    """创建PCA降维可视化"""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    fig = px.scatter(
        x=X_pca[:, 0], 
        y=X_pca[:, 1], 
        color=y.astype(str),
        title=f'PCA降维可视化 (解释方差: {pca.explained_variance_ratio_.sum():.3f})',
        labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.3f})',
                'y': f'PC2 ({pca.explained_variance_ratio_[1]:.3f})',
                'color': '标签'}
    )
    
    return fig

# === 项目管理函数 ===

def create_project_directory() -> str:
    """创建项目目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    project_name = f"{timestamp}_{random_id}"
    project_dir = os.path.join("./projects", project_name)
    os.makedirs(project_dir, exist_ok=True)
    return project_dir

def convert_numpy_types(obj):
    """将numpy类型转换为Python原生类型，以便JSON序列化"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def save_project_results(project_dir: str, results: Dict[str, Any], 
                        data_info: Dict[str, Any]) -> None:
    """保存项目结果"""
    # 保存模型性能结果
    performance_data = []
    for model_name, result in results.items():
        performance_data.append({
            'Model': model_name,
            'Accuracy': float(result['accuracy']) if result['accuracy'] is not None else None,
            'AUC': float(result['auc_score']) if result['auc_score'] is not None else None,
            'CV_Mean': float(result.get('cv_mean')) if result.get('cv_mean') is not None else None,
            'CV_Std': float(result.get('cv_std')) if result.get('cv_std') is not None else None
        })
    
    performance_df = pd.DataFrame(performance_data)
    performance_df.to_csv(os.path.join(project_dir, 'model_performance.csv'), index=False)
    
    # 保存最佳模型
    best_model_name = max(results.keys(), key=lambda k: results[k]['auc_score'])
    best_model = results[best_model_name]['model']
    
    try:
        joblib.dump(best_model, os.path.join(project_dir, 'best_model.pkl'))
    except Exception as e:
        st.error(f"保存最佳模型时出错: {e}")
    
    # 保存所有模型（可选）
    models_dir = os.path.join(project_dir, 'all_models')
    os.makedirs(models_dir, exist_ok=True)
    
    for model_name, result in results.items():
        try:
            model_filename = f"{model_name.replace(' ', '_')}_model.pkl"
            joblib.dump(result['model'], os.path.join(models_dir, model_filename))
        except Exception as e:
            st.warning(f"保存模型 {model_name} 时出错: {e}")
    
    # 保存预测结果（用于后续分析）
    try:
        predictions_data = {}
        for model_name, result in results.items():
            # 确保转换为Python原生列表类型
            y_pred = result['y_pred']
            y_prob = result['y_prob']
            
            if hasattr(y_pred, 'tolist'):
                predictions_data[f'{model_name}_pred'] = y_pred.tolist()
            else:
                predictions_data[f'{model_name}_pred'] = list(y_pred)
                
            if hasattr(y_prob, 'tolist'):
                predictions_data[f'{model_name}_prob'] = y_prob.tolist()
            else:
                predictions_data[f'{model_name}_prob'] = list(y_prob)
        
        predictions_df = pd.DataFrame(predictions_data)
        predictions_df.to_csv(os.path.join(project_dir, 'predictions.csv'), index=False)
    except Exception as e:
        st.warning(f"保存预测结果时出错: {e}")
    
    # 保存项目信息 - 确保所有数据类型都是JSON可序列化的
    project_info = {
        'creation_time': datetime.now().isoformat(),
        'best_model': best_model_name,
        'best_auc': float(results[best_model_name]['auc_score']),
        'best_accuracy': float(results[best_model_name]['accuracy']),
        'models_count': len(results),
        'models_list': list(results.keys()),
        'data_info': convert_numpy_types(data_info)  # 转换numpy类型
    }
    
    import json
    with open(os.path.join(project_dir, 'project_info.json'), 'w', encoding='utf-8') as f:
        json.dump(project_info, f, ensure_ascii=False, indent=2)

# === 主应用界面 ===

def show_sidebar():
    """显示侧边栏"""
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <h2 style="color: black;">🧬 CADD平台</h2>
        <p style="color: black;">计算机辅助药物设计</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 导航菜单 - 移除项目管理和高级分析
    menu_options = {
        "🏠 首页": "首页",
        "📊 数据展示": "数据展示", 
        "🤖 模型训练": "模型训练",
        "🔬 活性预测": "活性预测",
        "📚 知识获取": "知识获取"
    }
    
    selected = st.sidebar.selectbox(
        "选择功能模块",
        list(menu_options.keys()),
        format_func=lambda x: x
    )
    
    return menu_options[selected]

def show_homepage():
    """显示首页"""
    # 主标题
    st.markdown("""
    <div class="main-container">
        <h1>🧬 2025 CADD课程实践平台</h1>
        <p style="font-size: 1.2em;">现代化计算机辅助药物设计工具套件</p>
        <p>集成多种机器学习模型和交互式可视化</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 功能介绍卡片
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>📊 数据展示</h3>
            <p>• 多维度数据可视化</p>
            <p>• 相关性分析</p>
            <p>• PCA降维可视化</p>
            <p>• 分子描述符分析</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>🤖 模型训练</h3>
            <p>• 多模型对比训练</p>
            <p>• 自动超参数优化</p>
            <p>• 交叉验证评估</p>
            <p>• 性能可视化分析</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-card">
            <h3>🔬 活性预测</h3>
            <p>• 单分子预测</p>
            <p>• 批量预测分析</p>
            <p>• 预测置信度评估</p>
            <p>• 文献搜索功能</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 平台特色
    st.markdown("## 🚀 平台特色")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 技术优势
        - **多模型支持**: RF、SVM、XGB、LGB、NN
        - **交互式可视化**: Plotly动态图表
        - **高性能计算**: 并行处理和缓存优化
        - **模块化架构**: 易于扩展和维护
        """)
    
    with col2:
        st.markdown("""
        ### 功能亮点
        - **智能化分析**: 自动特征工程和模型选择
        - **分子可视化**: 化学结构编辑和展示
        - **文献获取**: 自动搜索相关研究论文
        - **批量处理**: 支持大规模数据预测
        """)

# 主程序入口
def main():
    """主程序"""
    # 显示侧边栏并获取选择的页面
    selected_page = show_sidebar()
    
    # 根据选择显示对应页面
    if selected_page == "首页":
        show_homepage()
    elif selected_page == "数据展示":
        show_data_analysis()
    elif selected_page == "模型训练":
        show_model_training()
    elif selected_page == "活性预测":
        show_activity_prediction()
    elif selected_page == "知识获取":
        show_knowledge_acquisition()

def show_data_analysis():
    """数据展示模块"""
    st.title("📊 数据展示与分析")
    
    # 文件上传区域
    st.markdown("### 📁 数据上传")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "选择CSV数据文件", 
            type=['csv'],
            help="上传包含SMILES和活性标签的CSV文件"
        )
    
    with col2:
        st.markdown("""
        **数据格式要求:**
        - CSV格式文件
        - 包含SMILES列
        - 包含活性标签列
        - 可包含其他特征列
        """)
    
    # 示例数据选择
    if not uploaded_file:
        st.markdown("### 📋 或选择示例数据")
        csv_files = glob.glob("./data/*.csv")
        if csv_files:
            example_file = st.selectbox(
                "选择示例数据集", 
                [os.path.basename(f) for f in csv_files]
            )
            selected_file = [f for f in csv_files if os.path.basename(f) == example_file][0]
            data = pd.read_csv(selected_file)
        else:
            st.warning("未找到示例数据文件，请上传数据或将CSV文件放入./data/目录")
            return
    else:
        data = pd.read_csv(uploaded_file)
    
    if 'data' in locals():
        # 数据基本信息
        st.markdown("### 📈 数据概况")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("样本总数", len(data))
        with col2:
            st.metric("特征数量", len(data.columns))
        with col3:
            st.metric("缺失值", data.isnull().sum().sum())
        with col4:
            if len(data.select_dtypes(include=[np.number]).columns) > 0:
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                st.metric("数值型特征", len(numeric_cols))
        
        # 数据预览
        st.markdown("### 🔍 数据预览")
        st.dataframe(data.head(10), use_container_width=True)
        
        # 数值型特征分析
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            st.markdown("### 📊 数值型特征分析")
            
            # 描述性统计
            st.markdown("#### 描述性统计")
            st.dataframe(data[numeric_cols].describe(), use_container_width=True)
            
            # 相关性分析
            if len(numeric_cols) > 1:
                st.markdown("#### 特征相关性分析")
                
                corr_matrix = data[numeric_cols].corr()
                
                # 创建交互式热力图
                fig = px.imshow(
                    corr_matrix,
                    labels=dict(x="特征", y="特征", color="相关系数"),
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    color_continuous_scale='RdBu_r',
                    aspect="auto",
                    title="特征相关性热力图"
                )
                
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
        
        # 分子相关分析（如果有SMILES列）
        smiles_cols = [col for col in data.columns if 'smiles' in col.lower()]
        if smiles_cols:
            st.markdown("### 🧪 分子分析")
            
            smiles_col = st.selectbox("选择SMILES列", smiles_cols)
            
            if st.button("计算分子描述符"):
                with st.spinner("正在计算分子描述符..."):
                    descriptors_list = []
                    
                    progress_bar = st.progress(0)
                    
                    for i, smiles in enumerate(data[smiles_col]):
                        if pd.notna(smiles):
                            descriptors = calculate_molecular_descriptors(smiles)
                            if descriptors:
                                descriptors_list.append(descriptors)
                        
                        progress_bar.progress((i + 1) / len(data))
                    
                    if descriptors_list:
                        descriptors_df = pd.DataFrame(descriptors_list)
                        
                        st.success(f"成功计算了 {len(descriptors_df)} 个分子的描述符")
                        
                        # 描述符统计
                        st.markdown("#### 分子描述符统计")
                        st.dataframe(descriptors_df.describe(), use_container_width=True)
                        
                        # Lipinski五规则检查
                        st.markdown("#### Lipinski五规则检查")
                        
                        lipinski_violations = []
                        for _, row in descriptors_df.iterrows():
                            violations = 0
                            if row.get('MolWt', 0) > 500:
                                violations += 1
                            if row.get('LogP', 0) > 5:
                                violations += 1
                            if row.get('NumHDonors', 0) > 5:
                                violations += 1
                            if row.get('NumHAcceptors', 0) > 10:
                                violations += 1
                            lipinski_violations.append(violations)
                        
                        violation_counts = pd.Series(lipinski_violations).value_counts().sort_index()
                        
                        fig = px.bar(
                            x=violation_counts.index,
                            y=violation_counts.values,
                            title='Lipinski五规则违反情况',
                            labels={'x': '违反规则数量', 'y': '分子数量'}
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 显示统计信息
                        drug_like = sum([v == 0 for v in lipinski_violations])
                        st.metric(
                            "药物样分子数量", 
                            f"{drug_like} / {len(lipinski_violations)} ({drug_like/len(lipinski_violations)*100:.1f}%)"
                        )

def show_model_training():
    st.title("🤖 多模型训练与比较")
    
    # 数据加载
    st.markdown("### 📁 数据准备")
    
    uploaded_file = st.file_uploader(
        "选择训练数据文件", 
        type=['csv'],
        key="training_data",
        help="上传包含SMILES和活性标签的CSV文件"
    )
    
    # 示例数据选择
    if not uploaded_file:
        csv_files = glob.glob("./data/*.csv")
        if csv_files:
            example_file = st.selectbox(
                "或选择示例数据集", 
                [os.path.basename(f) for f in csv_files],
                key="training_example"
            )
            selected_file = [f for f in csv_files if os.path.basename(f) == example_file][0]
            data = pd.read_csv(selected_file)
        else:
            st.warning("未找到数据文件，请上传训练数据")
            return
    else:
        data = pd.read_csv(uploaded_file)
    
    if 'data' in locals():
        st.success(f"数据加载成功！样本数量: {len(data)}, 特征数量: {len(data.columns)}")
        
        # 选择列
        st.markdown("### 🎯 特征和标签选择")
        
        col1, col2 = st.columns(2)
        
        with col1:
            smiles_cols = [col for col in data.columns if 'smiles' in col.lower()]
            smiles_col = st.selectbox("选择SMILES列", smiles_cols if smiles_cols else data.columns)
        
        with col2:
            label_cols = [col for col in data.columns if any(keyword in col.lower() for keyword in ['label', 'target', 'class', 'active'])]
            label_col = st.selectbox("选择标签列", label_cols if label_cols else data.columns)
        
        # 检查数据
        if smiles_col and label_col:
            # 检查标签分布
            label_counts = data[label_col].value_counts()
            st.markdown("#### 标签分布")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("正样本", int(label_counts.get(1, 0)))
            with col2:
                st.metric("负样本", int(label_counts.get(0, 0)))
            with col3:
                if len(label_counts) > 0:
                    ratio = min(label_counts) / max(label_counts)
                    st.metric("类别平衡比", f"{ratio:.3f}")
            
            # 特征工程
            st.markdown("### 🔧 特征工程")
            
            fingerprint_type = st.selectbox(
                "选择分子指纹类型",
                ["morgan", "maccs"],
                help="Morgan指纹更适合结构相似性"
            )
            
            if st.button("开始特征提取", type="primary"):
                with st.spinner("正在提取分子特征..."):
                    fingerprints = []
                    valid_indices = []
                    
                    progress_bar = st.progress(0)
                    
                    for i, smiles in enumerate(data[smiles_col]):
                        if pd.notna(smiles):
                            fp = mol_to_fingerprint(smiles, fingerprint_type)
                            if fp is not None:
                                fingerprints.append(fp)
                                valid_indices.append(i)
                        
                        progress_bar.progress((i + 1) / len(data))
                    
                    if fingerprints:
                        X = np.array(fingerprints)
                        y = data.iloc[valid_indices][label_col].values
                        
                        # 检查提取的特征和标签
                        st.markdown("#### 📋 特征提取结果")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("有效样本数", len(X))
                        with col2:
                            st.metric("特征维度", X.shape[1])
                        with col3:
                            nan_count = np.sum(np.isnan(y))
                            st.metric("标签中NaN数量", nan_count)
                        
                        # 检查标签数据质量
                        if np.any(np.isnan(y)):
                            st.warning(f"检测到标签中有 {np.sum(np.isnan(y))} 个NaN值，训练时将自动处理")
                        
                        # 检查特征数据质量
                        nan_features = np.sum(np.isnan(X).any(axis=1))
                        if nan_features > 0:
                            st.warning(f"检测到 {nan_features} 个样本的特征包含NaN值，训练时将自动处理")
                        
                        st.success(f"特征提取完成！准备训练数据...")
                        
                        # 存储到session state
                        st.session_state['X'] = X
                        st.session_state['y'] = y
                        st.session_state['fingerprint_type'] = fingerprint_type
                    else:
                        st.error("特征提取失败，请检查SMILES数据质量")
            
            # 模型训练
            if 'X' in st.session_state and 'y' in st.session_state:
                st.markdown("### 🎯 模型训练配置")
                
                X = st.session_state['X']
                y = st.session_state['y']
                
                # 模型选择
                config = load_config()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    selected_models = st.multiselect(
                        "选择要训练的模型",
                        list(config.models.keys()),
                        default=["随机森林", "XGBoost"],
                        help="建议选择2-4个模型进行比较"
                    )
                
                with col2:
                    test_size = st.slider("测试集比例", 0.1, 0.4, 0.2, 0.05)
                
                # 开始训练
                if st.button("🚀 开始训练", type="primary"):
                    if selected_models:
                        # 数据清理和验证
                        st.markdown("#### 🧹 数据清理")
                        X_clean, y_clean, cleaning_info = clean_data_for_training(X, y)
                        
                        # 验证清理后的数据
                        if validate_training_data(X_clean, y_clean):
                            try:
                                # 尝试分层抽样，如果失败则使用普通抽样
                                try:
                                    X_train, X_test, y_train, y_test = train_test_split(
                                        X_clean, y_clean, 
                                        test_size=test_size, 
                                        random_state=42, 
                                        stratify=y_clean
                                    )
                                except ValueError as e:
                                    st.warning(f"分层抽样失败，使用普通随机抽样: {e}")
                                    X_train, X_test, y_train, y_test = train_test_split(
                                        X_clean, y_clean, 
                                        test_size=test_size, 
                                        random_state=42
                                    )
                                
                                # 开始训练模型
                                results = train_multiple_models(
                                    X_train, X_test, y_train, y_test,
                                    selected_models
                                )
                            except Exception as e:
                                st.error(f"数据分割失败: {e}")
                                st.stop()
                        else:
                            st.error("数据验证失败，无法继续训练")
                            st.stop()
                        
                        # 自动保存训练结果
                        st.markdown("#### 💾 自动保存模型")
                        with st.spinner("正在保存训练结果..."):
                            project_dir = create_project_directory()
                            project_info = {
                                'fingerprint_type': fingerprint_type,
                                'test_size': float(test_size),  # 确保是float类型
                                'cleaning_info': convert_numpy_types(cleaning_info)  # 转换numpy类型
                            }
                            save_project_results(project_dir, results, project_info)
                            
                            # 同时保存预处理信息，用于预测时的一致性
                            preprocessing_info = {
                                'fingerprint_type': fingerprint_type,
                                'feature_shape': int(X_clean.shape[1]),  # 转换为int
                                'original_samples': int(cleaning_info['original_samples']),  # 转换为int
                                'final_samples': int(cleaning_info['final_samples'])  # 转换为int
                            }
                            
                            import json
                            with open(os.path.join(project_dir, 'preprocessing_info.json'), 'w', encoding='utf-8') as f:
                                json.dump(preprocessing_info, f, ensure_ascii=False, indent=2)
                        
                        st.success(f"✅ 模型已自动保存到: {project_dir}")
                        st.session_state['training_results'] = results
                        st.session_state['current_project_dir'] = project_dir
                        
                        # 显示性能比较
                        st.markdown("### 📊 模型性能比较")
                        
                        # 显示最佳模型信息
                        best_model_name = max(results.keys(), key=lambda k: results[k]['auc_score'])
                        best_auc = results[best_model_name]['auc_score']
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("🏆 最佳模型", best_model_name)
                        with col2:
                            st.metric("🎯 最佳AUC", f"{best_auc:.4f}")
                        with col3:
                            st.metric("📁 项目路径", os.path.basename(project_dir))
                        
                        # ROC曲线比较
                        roc_fig = create_roc_comparison_plot(results)
                        st.plotly_chart(roc_fig, use_container_width=True)
                        
                        # 模型比较图表
                        comparison_fig = create_model_comparison_plot(results)
                        st.plotly_chart(comparison_fig, use_container_width=True)
                        
                        # 额外保存选项
                        st.markdown("#### 📋 额外操作")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.button("🔍 跳转到活性预测"):
                                st.session_state['selected_project_for_prediction'] = os.path.basename(project_dir)
                                st.experimental_rerun()
                        
                        with col2:
                            if st.button("📊 查看项目详情"):
                                st.info(f"项目已保存，可在项目管理中查看详细信息")
                        
                        # 显示保存的文件列表
                        st.markdown("#### 📂 已保存文件")
                        saved_files = os.listdir(project_dir)
                        for file in saved_files:
                            file_path = os.path.join(project_dir, file)
                            file_size = os.path.getsize(file_path)
                            st.text(f"📄 {file} ({file_size} bytes)")
                    else:
                        st.warning("请至少选择一个模型进行训练")

def show_activity_prediction():
    st.title("🔬 分子活性预测")
    
    # 加载已训练的项目
    st.markdown("### 📂 选择预测模型")
    
    # 创建projects目录（如果不存在）
    os.makedirs('./projects', exist_ok=True)
    
    projects = glob.glob('./projects/*')
    if not projects:
        st.warning("未找到已训练的模型，请先训练模型")
        st.info("请前往 '🤖 模型训练' 页面创建模型")
        return
    
    # 获取项目信息并按时间排序
    project_info_list = []
    for project_path in projects:
        project_name = os.path.basename(project_path)
        info_file = os.path.join(project_path, 'project_info.json')
        
        if os.path.exists(info_file):
            try:
                import json
                with open(info_file, 'r', encoding='utf-8') as f:
                    info = json.load(f)
                
                project_info_list.append({
                    'name': project_name,
                    'path': project_path,
                    'creation_time': info.get('creation_time', ''),
                    'best_model': info.get('best_model', 'Unknown'),
                    'best_auc': info.get('best_auc', 0),
                    'valid': True
                })
            except:
                project_info_list.append({
                    'name': project_name,
                    'path': project_path,
                    'creation_time': '',
                    'best_model': 'Unknown',
                    'best_auc': 0,
                    'valid': False
                })
        else:
            project_info_list.append({
                'name': project_name,
                'path': project_path,
                'creation_time': '',
                'best_model': 'Unknown',
                'best_auc': 0,
                'valid': False
            })
    
    # 按创建时间排序（最新的在前）
    project_info_list.sort(key=lambda x: x['creation_time'], reverse=True)
    
    # 检查是否有从训练模块跳转过来的项目
    default_index = 0
    if 'selected_project_for_prediction' in st.session_state:
        target_project = st.session_state['selected_project_for_prediction']
        for i, project_info in enumerate(project_info_list):
            if project_info['name'] == target_project:
                default_index = i
                break
        # 清除session state
        del st.session_state['selected_project_for_prediction']
    
    # 显示项目选择
    if project_info_list:
        # 创建项目显示选项
        project_options = []
        for info in project_info_list:
            if info['valid']:
                option_text = f"🟢 {info['name']} | {info['best_model']} | AUC: {info['best_auc']:.4f}"
            else:
                option_text = f"🔴 {info['name']} | 信息不完整"
            project_options.append(option_text)
        
        selected_index = st.selectbox(
            "选择训练好的项目", 
            range(len(project_options)),
            format_func=lambda x: project_options[x],
            index=default_index
        )
        
        selected_project_info = project_info_list[selected_index]
        project_dir = selected_project_info['path']
        
        # 显示项目详细信息
        if selected_project_info['valid']:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🏆 最佳模型", selected_project_info['best_model'])
            with col2:
                st.metric("🎯 AUC得分", f"{selected_project_info['best_auc']:.4f}")
            with col3:
                creation_time = selected_project_info['creation_time'][:19] if selected_project_info['creation_time'] else "Unknown"
                st.metric("⏰ 创建时间", creation_time)
            with col4:
                # 检查模型文件
                model_file = os.path.join(project_dir, 'best_model.pkl')
                model_status = "✅ 可用" if os.path.exists(model_file) else "❌ 缺失"
                st.metric("📁 模型文件", model_status)
    else:
        st.error("无有效项目")
        return
    
    # 加载模型和预处理信息
    model_file = os.path.join(project_dir, 'best_model.pkl')
    preprocessing_file = os.path.join(project_dir, 'preprocessing_info.json')
    
    if not os.path.exists(model_file):
        st.error("❌ 模型文件不存在，请重新训练模型")
        return
    
    try:
        # 加载模型
        with st.spinner("正在加载模型..."):
            model = joblib.load(model_file)
        
        # 加载预处理信息
        preprocessing_info = {}
        if os.path.exists(preprocessing_file):
            try:
                import json
                with open(preprocessing_file, 'r', encoding='utf-8') as f:
                    preprocessing_info = json.load(f)
            except:
                st.warning("⚠️ 无法加载预处理信息，将使用默认设置")
                preprocessing_info = {'fingerprint_type': 'morgan'}
        else:
            st.warning("⚠️ 预处理信息文件不存在，将使用默认设置")
            preprocessing_info = {'fingerprint_type': 'morgan'}
        
        # 显示加载信息
        col1, col2 = st.columns(2)
        with col1:
            st.success("✅ 模型加载成功！")
        with col2:
            fingerprint_type = preprocessing_info.get('fingerprint_type', 'morgan')
            st.info(f"🧬 指纹类型: {fingerprint_type}")
        
    except Exception as e:
        st.error(f"❌ 模型加载失败: {e}")
        st.info("可能的原因：模型文件损坏或版本不兼容")
        return
    
    # 预测选项
    prediction_mode = st.radio(
        "选择预测模式",
        ["单分子预测", "批量预测"],
        horizontal=True
    )
    
    if prediction_mode == "单分子预测":
        st.markdown("### 🧪 单分子预测")
        
        # 分子输入
        smiles_input = st.text_input(
            "输入分子SMILES",
            value="CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
            help="输入要预测的分子的SMILES字符串"
        )
        
        # 分子编辑器
        smile_code = st_ketcher(smiles_input)
        st.markdown(f"**当前分子SMILES:** `{smile_code}`")
        
        if smile_code and st.button("🔍 开始预测", type="primary"):
            with st.spinner("正在进行预测..."):
                # 使用训练时相同的指纹类型
                fingerprint_type = preprocessing_info.get('fingerprint_type', 'morgan')
                fingerprint = mol_to_fingerprint(smile_code, fingerprint_type)
                
                if fingerprint is not None:
                    # 进行预测
                    try:
                        if hasattr(model, 'predict_proba'):
                            prediction_prob = model.predict_proba([fingerprint])[0]
                            prediction = model.predict([fingerprint])[0]
                            confidence = max(prediction_prob)
                            active_prob = prediction_prob[1]
                        else:
                            prediction_prob = model.predict([fingerprint])[0]
                            prediction = 1 if prediction_prob > 0.5 else 0
                            confidence = prediction_prob if prediction == 1 else (1 - prediction_prob)
                            active_prob = prediction_prob
                        
                        # 显示预测结果
                        st.markdown("### 📋 预测结果")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            result_text = "活性" if prediction == 1 else "非活性"
                            st.metric("预测结果", result_text)
                        
                        with col2:
                            st.metric("活性概率", f"{active_prob:.3f}")
                        
                        with col3:
                            st.metric("预测置信度", f"{confidence:.3f}")
                        
                        # 分子描述符
                        st.markdown("### 🧮 分子描述符")
                        descriptors = calculate_molecular_descriptors(smile_code)
                        if descriptors:
                            desc_df = pd.DataFrame([descriptors])
                            st.dataframe(desc_df, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"预测过程中出现错误: {e}")
                else:
                    st.error("无法解析该SMILES字符串，请检查输入")
    
    elif prediction_mode == "批量预测":
        st.markdown("### 📊 批量预测")
        
        # 文件上传
        uploaded_file = st.file_uploader(
            "上传包含SMILES的CSV文件",
            type=['csv'],
            help="CSV文件应包含一列SMILES数据"
        )
        
        if uploaded_file:
            batch_data = pd.read_csv(uploaded_file)
            st.write("数据预览:")
            st.dataframe(batch_data.head(), use_container_width=True)
            
            # 选择SMILES列
            smiles_columns = [col for col in batch_data.columns if 'smiles' in col.lower()]
            if not smiles_columns:
                smiles_columns = batch_data.columns.tolist()
            
            smiles_col = st.selectbox("选择SMILES列", smiles_columns)
            
            if st.button("🚀 开始批量预测", type="primary"):
                with st.spinner("正在进行批量预测..."):                        
                    predictions = []
                    probabilities = []
                    valid_indices = []
                    
                    progress_bar = st.progress(0)
                    
                    # 使用训练时相同的指纹类型
                    fingerprint_type = preprocessing_info.get('fingerprint_type', 'morgan')
                    
                    for i, smiles in enumerate(batch_data[smiles_col]):
                        if pd.notna(smiles):
                            fingerprint = mol_to_fingerprint(smiles, fingerprint_type)
                            if fingerprint is not None:
                                try:
                                    if hasattr(model, 'predict_proba'):
                                        pred_prob = model.predict_proba([fingerprint])[0]
                                        pred = model.predict([fingerprint])[0]
                                        prob = pred_prob[1]
                                    else:
                                        pred_prob = model.predict([fingerprint])[0]
                                        pred = 1 if pred_prob > 0.5 else 0
                                        prob = pred_prob
                                    
                                    predictions.append(pred)
                                    probabilities.append(prob)
                                    valid_indices.append(i)
                                except:
                                    pass
                        
                        progress_bar.progress((i + 1) / len(batch_data))
                    
                    if predictions:
                        # 创建结果数据框
                        results_data = batch_data.iloc[valid_indices].copy()
                        results_data['预测结果'] = predictions
                        results_data['活性概率'] = probabilities
                        results_data['预测标签'] = ['活性' if p == 1 else '非活性' for p in predictions]
                        
                        st.success(f"批量预测完成！成功预测 {len(predictions)} 个分子")
                        
                        # 显示预测统计
                        st.markdown("### 📈 预测统计")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("总预测数", len(predictions))
                        with col2:
                            active_count = sum(predictions)
                            st.metric("预测活性", active_count)
                        with col3:
                            inactive_count = len(predictions) - active_count
                            st.metric("预测非活性", inactive_count)
                        with col4:
                            avg_prob = np.mean(probabilities)
                            st.metric("平均活性概率", f"{avg_prob:.3f}")
                        
                        # 概率分布图
                        fig = px.histogram(
                            x=probabilities,
                            nbins=20,
                            title="活性概率分布",
                            labels={'x': '活性概率', 'y': '分子数量'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 显示结果表格
                        st.markdown("### 📋 预测结果")
                        st.dataframe(results_data, use_container_width=True)
                        
                        # 导出结果
                        csv_data = results_data.to_csv(index=False)
                        st.download_button(
                            label="📥 下载预测结果",
                            data=csv_data,
                            file_name=f"prediction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.error("批量预测失败，请检查SMILES数据质量")

def show_knowledge_acquisition():
    """知识获取与文献分析模块"""
    st.title("📚 知识获取与文献分析")
    
    # 设置Entrez邮箱
    Entrez.email = "cadd@example.com"
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_keyword = st.text_input(
            "输入搜索关键词",
            value="drug toxicity machine learning",
            help="输入英文关键词进行文献搜索"
        )
    
    with col2:
        max_results = st.selectbox(
            "最大搜索结果数",
            [5, 10, 15, 20],
            index=0
        )
    
    if st.button("🔍 搜索文献", type="primary"):
        with st.spinner("正在搜索文献..."):
            try:
                # 搜索PMC数据库
                handle = Entrez.esearch(
                    db="pubmed", 
                    term=search_keyword, 
                    retmode="xml", 
                    retmax=max_results
                )
                search_results = Entrez.read(handle)
                pmid_list = search_results["IdList"]
                
                if pmid_list:
                    st.success(f"找到 {len(pmid_list)} 篇相关文献")
                    
                    # 获取文献详细信息
                    handle = Entrez.efetch(
                        db="pubmed",
                        id=",".join(pmid_list),
                        rettype="abstract",
                        retmode="xml"
                    )
                    articles = Entrez.read(handle)
                    
                    # 存储文献信息到session state
                    st.session_state['articles'] = articles
                    st.session_state['pmid_list'] = pmid_list
                    
                    # 显示文献列表
                    st.markdown("### 📄 搜索结果")
                    
                    article_options = []
                    for i, article in enumerate(articles['PubmedArticle']):
                        try:
                            title = article['MedlineCitation']['Article']['ArticleTitle']
                            authors = article['MedlineCitation']['Article']['AuthorList']
                            if authors:
                                first_author = authors[0]['LastName'] + " " + authors[0].get('ForeName', '')
                            else:
                                first_author = "Unknown"
                            
                            # 限制标题长度
                            short_title = title[:100] + "..." if len(title) > 100 else title
                            article_options.append(f"{i+1}. {first_author} - {short_title}")
                        except:
                            article_options.append(f"{i+1}. 文献信息获取失败")
                    
                    # 让用户选择要查看abstract的文献
                    selected_articles = st.multiselect(
                        "选择要查看摘要的文献（可多选）",
                        range(len(article_options)),
                        format_func=lambda x: article_options[x],
                        default=list(range(min(3, len(article_options))))  # 默认选择前3篇
                    )
                    
                    # 显示选中文献的详细信息
                    if selected_articles:
                        st.markdown("### 📖 文献摘要")
                        
                        for idx in selected_articles:
                            article = articles['PubmedArticle'][idx]
                            
                            try:
                                # 提取文献信息
                                pmid = pmid_list[idx]
                                title = article['MedlineCitation']['Article']['ArticleTitle']
                                
                                # 作者信息
                                authors = article['MedlineCitation']['Article']['AuthorList']
                                author_names = []
                                for author in authors[:5]:  # 最多显示5个作者
                                    if 'LastName' in author and 'ForeName' in author:
                                        author_names.append(f"{author['LastName']} {author['ForeName']}")
                                
                                if len(authors) > 5:
                                    author_str = ", ".join(author_names) + ", et al."
                                else:
                                    author_str = ", ".join(author_names)
                                
                                # 期刊信息
                                journal = article['MedlineCitation']['Article']['Journal']['Title']
                                
                                # 发表日期
                                pub_date = article['MedlineCitation']['Article']['Journal']['JournalIssue']['PubDate']
                                year = pub_date.get('Year', 'Unknown')
                                
                                # 摘要
                                abstract_text = ""
                                if 'Abstract' in article['MedlineCitation']['Article']:
                                    abstract_list = article['MedlineCitation']['Article']['Abstract']['AbstractText']
                                    if isinstance(abstract_list, list):
                                        abstract_text = " ".join([str(abs_part) for abs_part in abstract_list])
                                    else:
                                        abstract_text = str(abstract_list)
                                
                                # 显示文献信息
                                with st.expander(f"📄 {title[:100]}{'...' if len(title) > 100 else ''}", expanded=True):
                                    st.markdown(f"**标题**: {title}")
                                    st.markdown(f"**作者**: {author_str}")
                                    st.markdown(f"**期刊**: {journal}")
                                    st.markdown(f"**年份**: {year}")
                                    st.markdown(f"**PMID**: {pmid}")
                                    
                                    if abstract_text:
                                        st.markdown("**摘要**:")
                                        st.write(abstract_text)
                                    else:
                                        st.write("*该文献没有可用的摘要*")
                                    
                                    # PubMed链接
                                    pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                                    st.markdown(f"🔗 [在PubMed中查看]({pubmed_url})")
                            
                            except Exception as e:
                                st.error(f"处理第{idx+1}篇文献时出错: {e}")
                
                else:
                    st.warning("未找到相关文献，请尝试其他关键词")
                
            except Exception as e:
                st.error(f"搜索失败: {e}")
                st.info("可能的原因：网络连接问题或NCBI服务暂时不可用")
    
    # 搜索建议
    st.markdown("### 💡 搜索建议")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **热门搜索关键词**:
        - "machine learning drug discovery"
        - "QSAR molecular toxicity"  
        - "deep learning ADMET"
        - "virtual screening compounds"
        - "molecular descriptors prediction"
        """)
    
    with col2:
        st.markdown("""
        **搜索技巧**:
        - 使用英文关键词
        - 组合多个相关术语
        - 使用引号包含精确短语
        - 尝试不同的同义词
        - 添加年份限制（如 "2020:2024[pdat]"）
        """)

if __name__ == "__main__":
    main()