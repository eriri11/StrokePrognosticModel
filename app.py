import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost
import sklearn


# 设置页面配置
st.set_page_config(
    page_title="脑卒中预后预测系统",
    page_icon="🩺",
    layout="wide"
)

# 应用标题
st.title("脑卒中预后预测系统")
st.markdown("使用机器学习模型预测脑卒中预后风险")


# 加载模型和预处理管道
@st.cache_data
def load_models():
    models = {}
    model_names = ["XGBoost", "RandomForest", "SVM", "LogisticRegression", "MLP"]

    for name in model_names:
        try:
            # 使用相对路径，假设模型文件与 app.py 在同一目录下的 models 文件夹中
            model_path = f"{name}_model.pkl"
            models[name] = joblib.load(model_path)
        except Exception as e:
            st.error(f"无法加载{name}模型: {str(e)}")

    return models


# 加载特征名称（从原始数据获取）
@st.cache_data
def load_feature_names():
    # 这里需要根据你的实际数据调整特征名称
    # 以下是示例特征名称，你需要替换为实际使用的特征
    numeric_features = [
        "onset", "DNT", "收缩压", "到院收缩压", "溶栓前肌力左上", "溶栓前肌力右下",
        "溶栓前braden", "溶栓前NIHSS", "溶栓结束时NIHSS", "血糖", "血糖2", "溶栓前-溶栓结束"
    ]

    categorical_features = [
        "溶栓后波利维", "溶栓后双抗"
    ]

    return numeric_features, categorical_features


# 主函数
def main():
    # 加载模型
    models = load_models()
    if not models:
        st.error("没有可用的模型，请确保模型文件已正确保存。")
        return

    # 加载特征名称
    numeric_features, categorical_features = load_feature_names()

    # 侧边栏：选择模型
    st.sidebar.header("模型选择")
    selected_model = st.sidebar.selectbox(
        "选择预测模型",
        list(models.keys())
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 关于")
    st.sidebar.info("这是一个基于机器学习的脑卒中预后预测系统，支持多种预测模型。")

    # 主界面：输入特征
    st.header("患者信息输入")

    # 创建输入表单
    with st.form("prediction_form"):
        st.subheader("数值型特征")
        input_data = {}

        # 数值型特征输入
        for feature in numeric_features:
            value = st.number_input(f"{feature}", value=0.0, step=0.1)
            input_data[feature] = value

        st.subheader("分类特征")
        # 分类特征输入
        for feature in categorical_features:
            options = ["否", "是"]
            value = st.selectbox(f"{feature}", options)
            input_data[feature] = 1 if value == "是" else 0

        # 提交按钮
        submitted = st.form_submit_button("预测")

        if submitted:
            # 转换输入数据为DataFrame
            input_df = pd.DataFrame([input_data])

            # 使用模型进行预测
            model = models[selected_model]

            try:
                # 进行预测
                if hasattr(model, "predict_proba"):
                    proba = 1 - model.predict_proba(input_df)[:, 1][0]
                else:
                    # 对于没有predict_proba的模型，使用decision_function
                    decision = model.decision_function(input_df)[0]
                    # 将decision转换为概率-like值
                    proba = 1 / (1 + np.exp(-decision))

                # 显示预测结果
                st.header("预测结果")
                st.write(f"**模型**: {selected_model}")
                st.write(f"**预测概率**: {proba:.4f}")

                # 根据概率给出建议
                if proba >= 0.7:
                    st.warning("高风险: 建议立即进行干预")
                elif proba >= 0.4:
                    st.info("中风险: 建议进一步检查")
                else:
                    st.success("低风险: 建议定期随访")

                # 显示决策曲线解释
                st.subheader("决策曲线解释")
                st.markdown("""
                决策曲线分析(DCA)帮助临床医生在不同阈值概率下选择最佳策略：
                - 当预测概率高于阈值时，建议进行干预
                - 当预测概率低于阈值时，建议不进行干预
                """)

            except Exception as e:
                st.error(f"预测出错: {str(e)}")


if __name__ == "__main__":
    main()
