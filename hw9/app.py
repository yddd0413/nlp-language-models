import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from transformers import pipeline


st.set_page_config(page_title="HW9 舆情分析仪表盘", layout="wide")


@st.cache_resource
def load_sentiment_pipeline():
    """
    使用轻量级多语言情感模型。
    模型标签通常为 positive/negative/neutral（大小写可能不同）。
    """
    model_name = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    return pipeline("sentiment-analysis", model=model_name)


def normalize_label(label: str) -> str:
    lower_label = label.lower()
    if "pos" in lower_label:
        return "Positive"
    if "neg" in lower_label:
        return "Negative"
    return "Neutral"


def analyze_text(text: str) -> Tuple[str, float]:
    analyzer = load_sentiment_pipeline()
    result = analyzer(text[:512])[0]
    return normalize_label(result["label"]), float(result["score"])


def build_gauge(label: str, score: float) -> go.Figure:
    color_map = {
        "Positive": "#22c55e",
        "Negative": "#ef4444",
        "Neutral": "#f59e0b",
    }
    color = color_map.get(label, "#60a5fa")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score * 100,
            number={"suffix": "%", "font": {"size": 40, "color": color}},
            gauge={
                "shape": "angular",
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
                "bar": {"color": color, "thickness": 0.35},
                "bgcolor": "#111827",
                "borderwidth": 1,
                "bordercolor": "#374151",
                "steps": [
                    {"range": [0, 40], "color": "#1f2937"},
                    {"range": [40, 70], "color": "#334155"},
                    {"range": [70, 100], "color": "#475569"},
                ],
            },
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": f"<b>{label}</b>", "font": {"size": 24, "color": color}},
        )
    )
    fig.update_layout(
        height=330,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="#0b1220",
        font={"color": "#e5e7eb"},
    )
    return fig


def generate_test_comments() -> List[str]:
    return [
        "这款手机拍照效果非常惊艳，夜景也很清晰。",
        "物流速度很快，包装完整，没有破损。",
        "电池续航一般，一天要充两次电。",
        "玩游戏半小时就发烫，体验不太好。",
        "屏幕显示细腻，色彩很舒服。",
        "客服回复很慢，问题一直没解决。",
        "价格有点贵，但整体做工不错。",
        "系统偶尔卡顿，不过重启后会恢复。",
        "音质很棒，看电影沉浸感不错。",
        "外观普通，没有宣传图那么好看。",
        "信号稳定，地铁里也能正常使用。",
        "更新系统后耗电变快了。",
        "键盘手感舒适，打字效率提升明显。",
        "摄像头对焦慢，抓拍容易糊。",
        "性价比高，推荐购买。",
    ]


def build_pie_chart(counts: Dict[str, int]) -> go.Figure:
    labels = ["Positive", "Negative", "Neutral"]
    values = [counts.get(k, 0) for k in labels]
    colors = ["#22c55e", "#ef4444", "#f59e0b"]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.45,
                marker=dict(colors=colors, line=dict(color="#0b1220", width=2)),
                textinfo="label+percent",
                textfont=dict(color="white", size=14),
            )
        ]
    )
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220",
        font=dict(color="#e5e7eb"),
        legend=dict(orientation="h", y=-0.1, x=0.25),
    )
    return fig


st.title("电商/社交媒体舆情分析工具")
st.caption("Streamlit + Hugging Face + Plotly")

tab1, tab2, tab3 = st.tabs(
    ["模块1：情感分类+置信度", "模块2：显式vs隐式情感", "模块3：舆情挖掘仪表盘"]
)

with tab1:
    st.subheader("基础情感分类与置信度量化")
    user_text = st.text_area(
        "输入中文商品评论",
        value="这款耳机音质非常好，降噪也很强，性价比很高！",
        height=120,
    )
    if st.button("分析单条评论情感", key="tab1_btn"):
        if not user_text.strip():
            st.warning("请输入评论文本。")
        else:
            with st.spinner("模型分析中，请稍候..."):
                try:
                    label, score = analyze_text(user_text)
                    c1, c2 = st.columns([1, 1.3])
                    with c1:
                        st.metric("情感类别", label)
                        st.metric("置信度", f"{score:.4f}")
                    with c2:
                        st.plotly_chart(build_gauge(label, score), use_container_width=True)
                except Exception as err:
                    st.error(f"分析失败：{err}")
    st.info("工程意义：分类结果告诉你“是什么情感”，置信度告诉你“模型有多确定”，低置信度样本往往需要人工复核。")

with tab2:
    st.subheader("显式情感 vs. 隐式情感识别")
    st.markdown(
        "- `显式情感`：带明显褒贬词，如“太棒了”“太垃圾了”。\n"
        "- `隐式情感`：表面是客观描述，但隐含态度，如“玩游戏半小时就没电了”。"
    )

    col1, col2 = st.columns(2)
    explicit_text = col1.text_area(
        "显式情感评价",
        value="这屏幕画质太垃圾了。",
        height=100,
    )
    implicit_text = col2.text_area(
        "隐式客观描述",
        value="在太阳底下根本看不清屏幕上的字。",
        height=100,
    )

    if st.button("对比分析两类表达", key="tab2_btn"):
        with st.spinner("正在识别情感..."):
            try:
                e_label, e_score = analyze_text(explicit_text)
                i_label, i_score = analyze_text(implicit_text)
                re1, re2 = st.columns(2)
                with re1:
                    st.markdown("#### 显式情感结果")
                    st.metric("类别", e_label)
                    st.metric("置信度", f"{e_score:.4f}")
                with re2:
                    st.markdown("#### 隐式情感结果")
                    st.metric("类别", i_label)
                    st.metric("置信度", f"{i_score:.4f}")
            except Exception as err:
                st.error(f"分析失败：{err}")
    st.warning("观察重点：小型模型通常对显式情感判断更准；对隐式负面语义（无明显情感词）可能出现漏判。")

with tab3:
    st.subheader("舆情挖掘与可视化仪表盘")
    st.markdown(
        """
        <style>
        .big-screen {
            background: linear-gradient(135deg, #0b1220, #111827);
            border: 1px solid #374151;
            border-radius: 14px;
            padding: 14px;
            color: #e5e7eb;
        }
        </style>
        <div class="big-screen"><b>Opinion Mining Dashboard</b>：批量评论 -> 情感统计 -> 宏观口碑洞察</div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("生成测试舆情数据", key="tab3_gen_btn"):
        st.session_state["opinion_data"] = generate_test_comments()
        random.shuffle(st.session_state["opinion_data"])

    comments = st.session_state.get("opinion_data", [])
    if comments:
        st.write("当前测试数据：")
        st.dataframe(pd.DataFrame({"评论文本": comments}), use_container_width=True, hide_index=True)

        if st.button("批量分析并生成图表", key="tab3_run_btn"):
            with st.spinner("批量分析中..."):
                try:
                    analyzer = load_sentiment_pipeline()
                    results = analyzer(comments)
                    labels = [normalize_label(r["label"]) for r in results]
                    counts = {
                        "Positive": labels.count("Positive"),
                        "Negative": labels.count("Negative"),
                        "Neutral": labels.count("Neutral"),
                    }
                    summary_df = pd.DataFrame(
                        {"情感类别": list(counts.keys()), "数量": list(counts.values())}
                    )
                    c1, c2 = st.columns([1, 1.2])
                    with c1:
                        st.dataframe(summary_df, use_container_width=True, hide_index=True)
                        st.metric("总评论数", str(len(comments)))
                    with c2:
                        st.plotly_chart(build_pie_chart(counts), use_container_width=True)
                    st.success("洞察：可用该分布追踪产品口碑变化，辅助产品改进与危机预警。")
                except Exception as err:
                    st.error(f"批量分析失败：{err}")
    else:
        st.info("请先点击“生成测试舆情数据”。")
