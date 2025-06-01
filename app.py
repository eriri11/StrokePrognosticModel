import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost
import sklearn
import shap
import matplotlib.pyplot as plt
from io import BytesIO
import matplotlib

# 设置全局中文字体 - 必须在其他matplotlib操作之前
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

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
            model_path = f"{name}_model.pkl"
            models[name] = joblib.load(model_path)
        except Exception as e:
            st.error(f"无法加载{name}模型: {str(e)}")

    return models


# 加载特征名称
@st.cache_data
def load_feature_names():
    numeric_features = [
        "onset", "DNT", "收缩压", "到院收缩压", "溶栓前肌力左上", "溶栓前肌力右下",
        "溶栓前braden", "溶栓前NIHSS", "溶栓结束时NIHSS", "血糖", "血糖2", "溶栓前-溶栓结束"
    ]
    categorical_features = ["溶栓后波利维", "溶栓后双抗"]
    return numeric_features, categorical_features


# 生成SHAP力导向图（Force Plot）
def generate_shap_force_plot(model, input_df, model_name, feature_names):
    try:
        # 为不同模型类型创建解释器
        if model_name in ["XGBoost", "RandomForest", "LogisticRegression"]:
            explainer = shap.Explainer(model, feature_names=feature_names)
        else:  # 处理SVM、MLP等模型（使用KernelExplainer，较慢）
            explainer = shap.KernelExplainer(
                model.predict_proba if hasattr(model, "predict_proba") else model.predict,
                input_df,
                feature_names=feature_names
            )

        # 计算SHAP值
        shap_values = explainer(input_df)

        # 创建力导向图
        plt.figure(figsize=(10, 6))
        shap.plots.force(shap_values[0], matplotlib=True, show=False)  # 使用matplotlib渲染

        # 修复中文显示：设置坐标轴标签字体
        for text in plt.gca().get_yticklabels():
            text.set_fontproperties(matplotlib.font_manager.FontProperties(family='SimHei'))

        # 调整布局防止截断
        plt.tight_layout()

        # 转换为图像
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close()
        buf.seek(0)

        return buf
    except Exception as e:
        st.error(f"生成SHAP解释图失败: {str(e)}")
        return None


# 主函数
def main():
    models = load_models()
    if not models:
        st.error("没有可用的模型，请确保模型文件已正确保存并位于当前目录")
        return

    numeric_features, categorical_features = load_feature_names()
    all_features = numeric_features + categorical_features  # 合并所有特征

    st.sidebar.header("模型选择")
    selected_model = st.sidebar.selectbox("选择预测模型", list(models.keys()))

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 系统说明")
    st.sidebar.info("本系统通过机器学习模型分析脑卒中患者数据，预测溶栓后NIHSS评分变化风险")

    with st.form("prediction_form"):
        st.header("患者数据输入")

        # 数值型特征输入
        st.subheader("数值型特征（请输入数值）")
        input_data = {}
        for feature in numeric_features:
            value = st.number_input(
                f"{feature}",
                value=0.0,
                step=0.1,
                format="%.1f",
                help=f"请输入患者的{feature}数值"
            )
            input_data[feature] = value

        # 分类特征输入
        st.subheader("分类特征（请选择是否）")
        for feature in categorical_features:
            value = st.selectbox(
                f"{feature}",
                options=["否", "是"],
                help=f"请选择患者是否{feature}"
            )
            input_data[feature] = 1 if value == "是" else 0

        # 提交按钮
        submitted = st.form_submit_button("生成预测报告", use_container_width=True)

        if submitted:
            input_df = pd.DataFrame([input_data])
            model = models[selected_model]

            try:
                # 预测逻辑
                if hasattr(model, "predict_proba"):
                    proba_risk = 1 - model.predict_proba(input_df)[:, 1][0]  # 风险概率（NIHSS上升）
                else:
                    decision = model.decision_function(input_df)[0]
                    proba_risk = 1 / (1 + np.exp(-decision))  # 转换为概率

                # 显示预测结果
                st.header("📊 预测结果")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "风险概率",
                        f"{proba_risk:.4f}",
                        delta=f"{'▲' if proba_risk > 0.5 else '▼'} 相对于基准风险"
                    )
                with col2:
                    st.metric(
                        "安全概率",
                        f"{1 - proba_risk:.4f}",
                        delta=f"{'▲' if (1 - proba_risk) > 0.5 else '▼'} 相对于基准安全率"
                    )

                # 风险等级建议
                st.subheader("⚠️ 风险等级")
                if proba_risk >= 0.7:
                    st.error("高风险！建议立即启动干预流程", icon="🚑")
                elif proba_risk >= 0.4:
                    st.warning("中风险！建议24小时内安排进一步影像学检查", icon="📸")
                else:
                    st.success("低风险！建议常规护理并密切监测", icon="✅")

                # 生成并显示SHAP解释图
                st.header("🔍 模型解释（SHAP力导向图）")
                st.markdown("""
                每个箭头表示一个特征对预测结果的影响：
                - **红色**：该特征值越高，风险越高（箭头向右）
                - **蓝色**：该特征值越高，风险越低（箭头向左）
                - **箭头长度**：特征影响的强度
                """)

                shap_plot = generate_shap_force_plot(
                    model=model,
                    input_df=input_df,
                    model_name=selected_model,
                    feature_names=all_features
                )

                if shap_plot:
                    st.image(shap_plot, use_container_width=True, caption="特征影响分析")
                else:
                    st.warning("该模型暂不支持可视化解释（建议使用XGBoost或随机森林模型）")

            except Exception as e:
                st.error(f"预测过程中出现错误: {str(e)}", icon="🚨")
                st.exception(e)  # 调试时显示完整堆栈跟踪（部署时建议移除）


if __name__ == "__main__":
    main()
