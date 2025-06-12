#!/usr/bin/env python3
"""
2025 CADD课程实践平台启动脚本
"""

import os
import sys
import subprocess
import platform

def check_dependencies():
    """检查依赖是否安装"""
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'sklearn',
        'plotly',
        'rdkit'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ 缺少以下依赖包:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n请运行以下命令安装依赖:")
        print("pip install -r requirements.txt")
        print("conda install -c conda-forge rdkit")
        return False
    
    return True

def create_directories():
    """创建必要的目录"""
    directories = ['data', 'projects']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ 创建目录: {directory}")

def check_port(port=8501):
    """检查端口是否被占用"""
    import socket
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        result = s.connect_ex(('localhost', port))
        return result != 0

def main():
    """主函数"""
    print("🧬 2025 CADD课程实践平台")
    print("=" * 50)
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        print("❌ Python版本过低，请使用Python 3.8+")
        return
    
    print(f"✅ Python版本: {platform.python_version()}")
    
    # 检查依赖
    print("\n📦 检查依赖包...")
    if not check_dependencies():
        return
    
    print("✅ 所有依赖包已安装")
    
    # 创建目录
    print("\n📁 创建必要目录...")
    create_directories()
    
    # 检查端口
    port = 8501
    if not check_port(port):
        print(f"⚠️  端口 {port} 已被占用，尝试使用其他端口...")
        port = 8502
    
    print(f"🚀 启动应用...")
    print(f"📱 应用地址: http://localhost:{port}")
    print("💡 按 Ctrl+C 停止应用")
    print("=" * 50)
    
    # 启动Streamlit应用
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'app.py',
            '--server.port', str(port),
            '--server.headless', 'true',
            '--server.enableCORS', 'false',
            '--server.enableXsrfProtection', 'false'
        ])
    except KeyboardInterrupt:
        print("\n👋 应用已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")

if __name__ == "__main__":
    main() 