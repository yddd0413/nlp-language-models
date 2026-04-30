import re
import streamlit as st
import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


st.set_page_config(page_title="机器翻译实验平台（HW8）", layout="wide")
st.title("机器翻译实验平台（HW8）")
st.caption("基于 Streamlit + Transformers + NLTK，包含 NMT 翻译、规则翻译对比、BLEU 自动评测。")


@st.cache_resource
def ensure_nltk_resource() -> None:
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")


@st.cache_resource
def load_nmt_pipeline():
    """
    优先按作业要求使用 pipeline。
    不同 transformers 版本对任务名支持不同，因此做多策略回退。
    """
    model_name = "Helsinki-NLP/opus-mt-en-zh"
    task_candidates = ["translation_en_to_zh", "translation", "any-to-any"]
    last_error = None
    for task in task_candidates:
        try:
            return pipeline(task, model=model_name)
        except Exception as err:
            last_error = err
    raise RuntimeError(f"pipeline 初始化失败: {last_error}")


@st.cache_resource
def load_nmt_direct_model():
    """
    当 pipeline 任务名不兼容时，回退到直接 model.generate，保证可用性。
    """
    model_name = "Helsinki-NLP/opus-mt-en-zh"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


def nmt_translate(text: str) -> str:
    """
    统一翻译入口：
    1) 优先 pipeline（满足课程要求）
    2) 失败则回退 direct generate
    """
    try:
        translator = load_nmt_pipeline()
        result = translator(text, max_length=256)
        # 不同 task 的返回字段可能不同
        if isinstance(result, list) and result:
            item = result[0]
            return item.get("translation_text") or item.get("generated_text") or str(item)
        return str(result)
    except Exception:
        tokenizer, model = load_nmt_direct_model()
        inputs = tokenizer([text], return_tensors="pt", truncation=True)
        output_ids = model.generate(**inputs, max_length=256)
        return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]


def rule_based_translate(text: str) -> str:
    """
    简化版“基于规则/词典”的逐词直译。
    - 按空格和标点粗分词
    - 词典外词汇保留原词
    """
    lexicon = {
        "it": "它",
        "rains": "下雨",
        "cats": "猫",
        "and": "和",
        "dogs": "狗",
        "i": "我",
        "you": "你",
        "he": "他",
        "she": "她",
        "we": "我们",
        "they": "他们",
        "is": "是",
        "are": "是",
        "was": "是",
        "were": "是",
        "have": "有",
        "has": "有",
        "go": "去",
        "went": "去了",
        "to": "到",
        "school": "学校",
        "the": "这",
        "a": "一个",
        "an": "一个",
        "in": "在",
        "on": "在",
        "for": "为了",
        "with": "和",
        "because": "因为",
        "weather": "天气",
        "very": "非常",
        "good": "好",
        "bad": "坏",
        "today": "今天",
        "tomorrow": "明天",
        "founded": "创立",
        "apple": "苹果公司",
    }

    tokens = re.findall(r"[A-Za-z']+|[.,!?;:]", text)
    translated_tokens = []
    for token in tokens:
        lower_token = token.lower()
        translated_tokens.append(lexicon.get(lower_token, token))

    # 简单规则：标点前不加空格
    output = " ".join(translated_tokens)
    output = re.sub(r"\s+([.,!?;:])", r"\1", output)
    return output


def tokenize_for_bleu_zh(text: str) -> list[str]:
    """
    中文 BLEU 计算时，使用“字符级”切分，减少分词器依赖。
    """
    text = text.strip()
    return [ch for ch in text if ch.strip()]


ensure_nltk_resource()

tab1, tab2, tab3 = st.tabs(
    ["模块一：神经机器翻译 (NMT)", "模块二：规则翻译对比", "模块三：BLEU 自动评测"]
)


with tab1:
    st.subheader("模块一：神经机器翻译（英文 -> 中文）")
    st.write("输入英文句子，使用 Hugging Face `Helsinki-NLP/opus-mt-en-zh` 生成中文译文。")

    en_input = st.text_area(
        "请输入英文文本",
        value="It rains cats and dogs.",
        height=120,
        key="tab1_en_input",
    )

    if st.button("开始神经翻译", key="tab1_translate_button"):
        if not en_input.strip():
            st.warning("请输入英文文本后再翻译。")
        else:
            with st.spinner("模型加载与翻译中，请稍候..."):
                try:
                    zh_output = nmt_translate(en_input)
                    st.session_state["latest_source"] = en_input
                    st.session_state["latest_nmt_translation"] = zh_output
                    st.success("翻译完成")
                    st.text_area("NMT 中文译文", value=zh_output, height=120, key="tab1_zh_output")
                except Exception as err:
                    st.error(f"翻译失败：{err}")

    st.info(
        "观察建议：输入俚语或复杂句（如 `It rains cats and dogs.`），比较模型是否能处理语境，"
        "而不是机械地按字面翻译。"
    )


with tab2:
    st.subheader("模块二：NMT 与规则翻译对比")
    st.write("同一英文输入，左侧显示 NMT 输出，右侧显示基于词典的逐词直译结果。")

    compare_input = st.text_area(
        "请输入用于对比的英文句子",
        value=st.session_state.get("latest_source", "It rains cats and dogs."),
        height=120,
        key="tab2_en_input",
    )

    if st.button("执行对比翻译", key="tab2_compare_button"):
        if not compare_input.strip():
            st.warning("请输入英文文本后再执行。")
        else:
            with st.spinner("正在生成对比结果..."):
                nmt_output = ""
                nmt_error = ""
                try:
                    nmt_output = nmt_translate(compare_input)
                    st.session_state["latest_source"] = compare_input
                    st.session_state["latest_nmt_translation"] = nmt_output
                except Exception as err:
                    nmt_error = str(err)

                rbmt_output = rule_based_translate(compare_input)
                st.session_state["latest_rule_translation"] = rbmt_output

            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown("#### 神经机器翻译 (NMT)")
                if nmt_error:
                    st.error(f"NMT 出错：{nmt_error}")
                else:
                    st.text_area("NMT 结果", value=nmt_output, height=180, key="tab2_nmt_output")

            with col_right:
                st.markdown("#### 基于规则逐词直译")
                st.text_area("规则翻译结果", value=rbmt_output, height=180, key="tab2_rule_output")

    st.warning(
        "思考提示：规则翻译在处理语序倒装、定语从句、一词多义时，"
        "往往只能逐词匹配，难以理解全句上下文。"
    )


with tab3:
    st.subheader("模块三：BLEU 自动评测")
    st.write("输入原文、参考译文、候选译文，使用 NLTK 计算 BLEU 分数。")

    bleu_src = st.text_area(
        "1) 待翻译英文原文",
        value=st.session_state.get("latest_source", "It rains cats and dogs."),
        height=100,
        key="tab3_src",
    )
    ref_zh = st.text_area(
        "2) 中文参考译文 (Reference)",
        value="外面下着倾盆大雨。",
        height=100,
        key="tab3_ref",
    )
    cand_zh = st.text_area(
        "3) 机器候选译文 (Candidate)",
        value=st.session_state.get("latest_nmt_translation", ""),
        height=100,
        key="tab3_cand",
    )

    auto_fill = st.checkbox("使用模块一最新 NMT 结果填充 Candidate", value=False, key="tab3_autofill")
    if auto_fill:
        cand_zh = st.session_state.get("latest_nmt_translation", cand_zh)
        st.text_area("自动填充后的 Candidate", value=cand_zh, height=100, key="tab3_cand_preview")

    if st.button("计算 BLEU 分数", key="tab3_bleu_btn"):
        if not ref_zh.strip() or not cand_zh.strip():
            st.warning("Reference 和 Candidate 不能为空。")
        else:
            reference_tokens = tokenize_for_bleu_zh(ref_zh)
            candidate_tokens = tokenize_for_bleu_zh(cand_zh)
            smoother = SmoothingFunction().method1
            bleu = sentence_bleu(
                [reference_tokens],
                candidate_tokens,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smoother,
            )
            st.metric("BLEU 得分", f"{bleu:.4f}")

            if bleu >= 0.7:
                st.success("解释：BLEU 较高，候选译文与参考译文在 n-gram 匹配上比较接近。")
            elif bleu >= 0.4:
                st.info("解释：BLEU 中等，部分短语匹配较好，但仍有明显差异。")
            else:
                st.error("解释：BLEU 偏低，候选译文与参考译文在词序/词汇匹配上差异较大。")

    st.info(
        "实验建议：对同一句英文，分别提供“词汇一致但语序错误”的参考译文，"
        "以及“语序正确但同义替换”的参考译文，比较 BLEU 波动，分析其优缺点。"
    )
