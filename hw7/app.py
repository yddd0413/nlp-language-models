import streamlit as st
import spacy
from spacy import displacy
import pandas as pd
import json
import streamlit.components.v1 as components

# --- Setup & Configuration ---
st.set_page_config(page_title="知识图谱抽取与可视化系统", layout="wide")

@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except Exception as e:
        st.error(f"Failed to load spaCy model: {e}")
        return None

nlp = load_spacy_model()

st.title("知识图谱抽取与可视化交互系统 🧠")
st.markdown("集成 **实体识别 (NER)**、**关系抽取 (RE)** 与 **知识图谱可视化 (KG Visualization)** 的完整抽取链路。")

# --- Default Mock Text ---
default_text = """Steve Jobs was the co-founder of Apple Inc. He was born in San Francisco, California. 
Apple is headquartered in Cupertino. Steve Jobs also founded NeXT and acquired Pixar. 
Tim Cook is the current CEO of Apple."""

# --- Helper Functions ---
def extract_relations(doc, text):
    """
    关系抽取函数。
    结合 spaCy 的依赖句法分析提取简单的 SVO 关系。
    同时为了演示效果，针对特定词汇（如 Steve Jobs, Apple）提供精准的 Mock 关系。
    """
    relations = []
    
    # 1. 基础 Mock 关系库（保证核心实体的展示效果）
    mock_rules = [
        {"source": "Steve Jobs", "target": "Apple Inc.", "relation": "FOUNDER_OF"},
        {"source": "Steve Jobs", "target": "San Francisco", "relation": "BORN_IN"},
        {"source": "Apple Inc.", "target": "Cupertino", "relation": "HEADQUARTERED_IN"},
        {"source": "Steve Jobs", "target": "NeXT", "relation": "FOUNDER_OF"},
        {"source": "Steve Jobs", "target": "Pixar", "relation": "ACQUIRED"},
        {"source": "Tim Cook", "target": "Apple", "relation": "CEO_OF"},
    ]
    
    # 检查文本中是否包含这些 Mock 实体
    for rule in mock_rules:
        if rule["source"] in text and (rule["target"] in text or rule["target"].replace(" Inc.", "") in text):
            relations.append(rule)
            
    # 2. 动态抽取：基于 spaCy 句法依存的简易 SVO 抽取
    for token in doc:
        if token.pos_ == "VERB":
            # 找主语
            subj = [w for w in token.lefts if w.dep_ in ('nsubj', 'nsubjpass')]
            # 找宾语
            obj = [w for w in token.rights if w.dep_ in ('dobj', 'pobj', 'attr')]
            
            if subj and obj:
                s = subj[0]
                o = obj[0]
                
                # 扩展到完整实体名
                s_text = s.text
                o_text = o.text
                for ent in doc.ents:
                    if s in ent: s_text = ent.text
                    if o in ent: o_text = ent.text
                
                # 避免与 mock 规则重复
                is_duplicate = False
                for r in relations:
                    if r["source"] == s_text and r["target"] == o_text:
                        is_duplicate = True
                        break
                        
                if not is_duplicate and s_text != o_text:
                    relations.append({
                        "source": s_text,
                        "target": o_text,
                        "relation": token.lemma_.upper()
                    })
                    
    return relations

# --- UI Layout ---
st.header("1. 文本输入与实体高亮识别 (NER)")

text_input = st.text_area("请输入或粘贴一段英文语料：", default_text, height=120)

if text_input and nlp:
    doc = nlp(text_input)
    
    # 模块一：查看底层标注 Checkbox
    view_bio = st.checkbox("查看底层标注 (BIO 序列模式)", value=False)
    
    st.markdown("#### 实体识别结果")
    if view_bio:
        # 生成 BIO 序列纯文本
        bio_tokens = []
        for token in doc:
            if token.ent_iob_ == "O":
                bio_tokens.append(f"{token.text}/O")
            else:
                bio_tokens.append(f"{token.text}/{token.ent_iob_}-{token.ent_type_}")
                
        st.info(" ".join(bio_tokens))
    else:
        # 使用 spaCy displaCy 高亮渲染
        # 预设不同类别的颜色
        colors = {"PERSON": "#ffcccb", "ORG": "#cce5ff", "GPE": "#d4edda", "LOC": "#d4edda", "DATE": "#fff3cd"}
        options = {"ents": ["PERSON", "ORG", "GPE", "LOC", "DATE"], "colors": colors}
        
        html = displacy.render(doc, style="ent", options=options, jupyter=False)
        st.markdown(f'<div style="border: 1px solid #ddd; padding: 15px; border-radius: 5px; background-color: #fff;">{html}</div>', unsafe_allow_html=True)

    st.divider()
    
    # --- 模块二：关系抽取 (RE) ---
    st.header("2. 关系抽取面板 (Relation Extraction)")
    
    relations = extract_relations(doc, text_input)
    
    if not relations:
        st.warning("未能从当前文本中抽取出明显的关系对。")
    else:
        df_relations = pd.DataFrame(relations)
        df_relations.columns = ["主体 (Subject)", "客体 (Object)", "关系词 (Predicate)"]
        st.table(df_relations)
        
    st.divider()
    
    # --- 模块三：知识图谱可视化 (KG Visualization) ---
    st.header("3. 知识图谱交互式可视化 (Vis-network)")
    st.markdown("图谱支持：🖱️ **鼠标拖拽节点** | ⚙️ **滚轮缩放** | 📌 **固定布局（不自动抖动）**")
    
    if relations:
        # 转换 nodes 和 edges
        nodes = []
        edges = []
        node_ids = {}
        
        # 颜色映射
        color_map = {
            "PERSON": "#ffcccb",
            "ORG": "#cce5ff",
            "GPE": "#d4edda",
            "LOC": "#d4edda",
            "DEFAULT": "#e2e2e2"
        }
        
        # 1. 优先将实体加入 nodes (为了准确的类别颜色)
        for ent in doc.ents:
            if ent.text not in node_ids:
                node_id = len(node_ids) + 1
                node_ids[ent.text] = node_id
                color = color_map.get(ent.label_, color_map["DEFAULT"])
                nodes.append({
                    "id": node_id, 
                    "label": ent.text, 
                    "color": color, 
                    "group": ent.label_,
                    "shape": "dot"
                })
                
        # 2. 将关系中的节点加入 (可能有些实体没被 spaCy 识别到)
        for rel in relations:
            for entity in [rel["source"], rel["target"]]:
                if entity not in node_ids:
                    node_id = len(node_ids) + 1
                    node_ids[entity] = node_id
                    nodes.append({
                        "id": node_id, 
                        "label": entity, 
                        "color": color_map["DEFAULT"],
                        "shape": "dot"
                    })
                    
            # 添加边
            edges.append({
                "from": node_ids[rel["source"]],
                "to": node_ids[rel["target"]],
                "label": rel["relation"],
                "arrows": "to"
            })

        # 构建 vis-network 的 HTML 代码
        vis_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js"></script>
            <style type="text/css">
                #mynetwork {{
                    width: 100%;
                    height: 600px;
                    border: 1px solid lightgray;
                    border-radius: 10px;
                    background-color: #fcfcfc;
                }}
            </style>
        </head>
        <body>
        <div id="mynetwork"></div>
        <script type="text/javascript">
            // Parse data from Python
            var nodesData = {json.dumps(nodes)};
            var edgesData = {json.dumps(edges)};
            
            var nodes = new vis.DataSet(nodesData);
            var edges = new vis.DataSet(edgesData);

            var container = document.getElementById('mynetwork');
            var data = {{
                nodes: nodes,
                edges: edges
            }};
            var options = {{
                layout: {{
                    randomSeed: 42
                }},
                nodes: {{
                    size: 25,
                    font: {{ size: 16, face: 'Arial' }},
                    borderWidth: 2
                }},
                edges: {{
                    font: {{ size: 12, align: 'middle' }},
                    color: {{ color: '#848484', highlight: '#2B7CE9' }},
                    smooth: {{ type: 'continuous' }}
                }},
                physics: {{
                    enabled: false
                }},
                interaction: {{
                    zoomView: true,
                    dragNodes: true,
                    dragView: true,
                    hover: true,
                    tooltipDelay: 200
                }}
            }};
            var network = new vis.Network(container, data, options);
        </script>
        </body>
        </html>
        """
        
        # 在 Streamlit 中渲染 HTML 组件
        components.html(vis_html, height=620)
    else:
        st.info("没有足够的关系数据来生成知识图谱。")
