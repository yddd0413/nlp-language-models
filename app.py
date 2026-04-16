import streamlit as st
import nltk
from nltk.util import ngrams
from nltk.lm.preprocessing import padded_everygram_pipeline, pad_both_ends
from nltk.lm import MLE, Laplace
from nltk.tokenize import word_tokenize, sent_tokenize
import torch
import torch.nn as nn
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
import numpy as np

# Page Config
st.set_page_config(page_title="NLP Language Models", layout="wide")

# Ensure NLTK data is downloaded
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')

download_nltk_data()

st.title("NLP Language Models Web App")

# Create Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "N-gram Language Model", 
    "RNN Language Model", 
    "BERT vs GPT-2", 
    "GPT-2 Perplexity"
])

# --- Tab 1: N-gram Language Model ---
with tab1:
    st.header("基于统计的 N 元语言模型")
    st.markdown("使用 NLTK 构建 Trigram 模型，并计算句子的生成概率（支持加一平滑）。")
    
    corpus_input = st.text_area(
        "输入基础英文语料", 
        "The quick brown fox jumps over the lazy dog. The dog is lazy but the fox is quick.", 
        height=150
    )
    
    sentence_input = st.text_input("输入要计算概率的句子", "The quick brown fox")
    
    use_smoothing = st.checkbox("开启加一平滑 (Add-one / Laplace Smoothing)", value=False)
    
    if st.button("构建模型并计算概率"):
        if not corpus_input or not sentence_input:
            st.error("请输入语料和测试句子！")
        else:
            with st.spinner("正在构建 Trigram 模型..."):
                # 预处理语料
                sents = sent_tokenize(corpus_input.lower())
                tokenized_text = [word_tokenize(s) for s in sents]
                
                n = 3 # Trigram
                
                # 训练 MLE 模型 (无平滑)
                train_data_mle, vocab_mle = padded_everygram_pipeline(n, tokenized_text)
                mle_model = MLE(n)
                mle_model.fit(train_data_mle, vocab_mle)
                
                # 训练 Laplace 模型 (加一平滑)
                train_data_lap, vocab_lap = padded_everygram_pipeline(n, tokenized_text)
                laplace_model = Laplace(n)
                laplace_model.fit(train_data_lap, vocab_lap)
                
                # 预处理测试句子
                test_tokens = word_tokenize(sentence_input.lower())
                test_padded = list(pad_both_ends(test_tokens, n=n))
                test_ngrams = list(ngrams(test_padded, n))
                
                st.write("**句子切分后的 Trigrams:**")
                st.write(test_ngrams)
                
                # 计算概率
                mle_prob = 1.0
                laplace_prob = 1.0
                has_zero_prob = False
                
                mle_probs_list = []
                laplace_probs_list = []
                
                for ngram in test_ngrams:
                    context = ngram[:-1]
                    word = ngram[-1]
                    
                    # MLE
                    p_mle = mle_model.score(word, context)
                    mle_prob *= p_mle
                    mle_probs_list.append(p_mle)
                    if p_mle == 0:
                        has_zero_prob = True
                        
                    # Laplace
                    p_lap = laplace_model.score(word, context)
                    laplace_prob *= p_lap
                    laplace_probs_list.append(p_lap)
                
                st.subheader("计算结果")
                
                if has_zero_prob:
                    st.warning("⚠️ 输入句子中包含语料库未出现的 Trigram（零概率事件）！")
                    st.markdown("展示平滑前后的概率对比：")
                    
                    col1, col2 = st.columns(2)
                    col1.metric("MLE (平滑前) 联合概率", f"{mle_prob:.4e}")
                    col2.metric("Laplace (平滑后) 联合概率", f"{laplace_prob:.4e}")
                    
                    # 详细对比
                    df_compare = pd.DataFrame({
                        "Trigram": [str(ng) for ng in test_ngrams],
                        "MLE 概率": mle_probs_list,
                        "Laplace 概率": laplace_probs_list
                    })
                    st.dataframe(df_compare)
                else:
                    if use_smoothing:
                        st.success(f"当前使用的模型（加一平滑）计算得到的联合概率为: **{laplace_prob:.4e}**")
                    else:
                        st.success(f"当前使用的模型（无平滑）计算得到的联合概率为: **{mle_prob:.4e}**")


# --- Tab 2: RNN Language Model ---
with tab2:
    st.header("RNN 语言模型 (字符级)")
    st.markdown("使用 PyTorch 训练一个 Character-level RNN 并生成文本。")
    
    rnn_corpus = st.text_area(
        "输入自定义短语料（如诗歌或名言）", 
        "To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer The slings and arrows of outrageous fortune, Or to take arms against a sea of troubles, And by opposing end them?", 
        height=150
    )
    
    col1, col2, col3 = st.columns(3)
    hidden_size = col1.slider("Hidden Size (隐藏层维度)", 16, 128, 64, step=16)
    epochs = col2.slider("Epochs (训练轮数)", 10, 500, 100, step=10)
    learning_rate = col3.selectbox("Learning Rate (学习率)", [0.001, 0.005, 0.01, 0.05, 0.1], index=2)
    
    # Define RNN Model
    class CharRNN(nn.Module):
        def __init__(self, vocab_size, hidden_size):
            super(CharRNN, self).__init__()
            self.hidden_size = hidden_size
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, vocab_size)
            
        def forward(self, x, hidden=None):
            x = self.embedding(x)
            out, hidden = self.rnn(x, hidden)
            out = self.fc(out.reshape(-1, self.hidden_size))
            return out, hidden

    if st.button("开始训练"):
        if not rnn_corpus:
            st.error("请输入语料！")
        else:
            chars = sorted(list(set(rnn_corpus)))
            vocab_size = len(chars)
            char_to_ix = {ch: i for i, ch in enumerate(chars)}
            ix_to_char = {i: ch for i, ch in enumerate(chars)}
            
            st.session_state.chars = chars
            st.session_state.char_to_ix = char_to_ix
            st.session_state.ix_to_char = ix_to_char
            
            # Prepare data
            data = [char_to_ix[ch] for ch in rnn_corpus]
            X = torch.tensor(data[:-1]).unsqueeze(0) # (1, seq_len)
            Y = torch.tensor(data[1:]) # (seq_len)
            
            model = CharRNN(vocab_size, hidden_size)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            
            st.write("训练中...")
            progress_bar = st.progress(0)
            loss_chart_placeholder = st.empty()
            
            losses = []
            
            for epoch in range(epochs):
                optimizer.zero_grad()
                output, _ = model(X)
                loss = criterion(output, Y)
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
                
                # Update chart and progress every 10% of epochs or at the end
                update_interval = max(1, epochs // 10)
                if (epoch + 1) % update_interval == 0 or epoch == epochs - 1:
                    loss_chart_placeholder.line_chart(losses)
                    progress_bar.progress((epoch + 1) / epochs)
            
            st.success("训练完成！Loss 曲线如上图所示。")
            st.session_state.rnn_model = model
            
    st.divider()
    st.subheader("文本生成")
    
    col_seed, col_len = st.columns(2)
    seed = col_seed.text_input("起始字符 (Seed)", "T")
    gen_length = col_len.slider("生成长度 (字符数)", 10, 200, 50, step=10)
    
    if st.button("生成文本"):
        if 'rnn_model' not in st.session_state:
            st.warning("请先在上方进行模型训练！")
        elif not seed:
            st.warning("请输入起始字符！")
        else:
            model = st.session_state.rnn_model
            char_to_ix = st.session_state.char_to_ix
            ix_to_char = st.session_state.ix_to_char
            
            model.eval()
            with torch.no_grad():
                # 处理未见过的字符
                input_seq = [char_to_ix.get(ch, 0) for ch in seed]
                input_tensor = torch.tensor(input_seq).unsqueeze(0) # (1, seq_len)
                
                generated = seed
                hidden = None
                
                # Warm up
                for i in range(len(seed) - 1):
                    _, hidden = model(input_tensor[:, i].unsqueeze(1), hidden)
                    
                curr_input = input_tensor[:, -1].unsqueeze(1)
                
                for _ in range(gen_length):
                    output, hidden = model(curr_input, hidden)
                    
                    # 采样
                    prob = torch.softmax(output[-1], dim=0).numpy()
                    next_char_ix = np.random.choice(len(prob), p=prob)
                    next_char = ix_to_char[next_char_ix]
                    
                    generated += next_char
                    curr_input = torch.tensor([[next_char_ix]])
                    
            st.info(f"**生成结果:**\n\n{generated}")


# --- Tab 3: BERT vs GPT-2 ---
with tab3:
    st.header("生成机制对比: BERT vs GPT-2")
    
    @st.cache_resource
    def load_pipelines():
        # Load BERT
        bert_pipe = pipeline('fill-mask', model='bert-base-uncased')
        # Load GPT-2
        gpt2_pipe = pipeline('text-generation', model='gpt2')
        return bert_pipe, gpt2_pipe
    
    with st.spinner("正在加载预训练模型（首次加载可能需要一些时间）..."):
        try:
            bert_pipe, gpt2_pipe = load_pipelines()
            models_loaded = True
        except Exception as e:
            st.error(f"模型加载失败: {e}")
            models_loaded = False
            
    if models_loaded:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("BERT (Masked Language Modeling)")
            st.markdown("填空式生成：根据上下文预测被遮挡的词 [MASK]。")
            bert_input = st.text_input("输入带有 [MASK] 标记的句子", "The man went to the [MASK] to buy some milk.")
            
            if st.button("BERT 预测"):
                if "[MASK]" not in bert_input:
                    st.error("句子中必须包含 '[MASK]' 标记。")
                else:
                    with st.spinner("BERT 预测中..."):
                        results = bert_pipe(bert_input, top_k=5)
                        
                        df_bert = pd.DataFrame({
                            "预测词 (Token)": [res['token_str'] for res in results],
                            "概率 (Score)": [f"{res['score']:.4f}" for res in results]
                        })
                        st.table(df_bert)
                        
        with col2:
            st.subheader("GPT-2 (Causal Language Modeling)")
            st.markdown("自回归生成：根据前缀提示词，从左到右预测后续词。")
            gpt2_input = st.text_input("输入前缀提示词 (Prompt)", "The man went to the")
            
            if st.button("GPT-2 生成"):
                if not gpt2_input:
                    st.error("请输入提示词。")
                else:
                    with st.spinner("GPT-2 生成中..."):
                        # pad_token_id=50256 is eos_token_id for GPT2
                        results = gpt2_pipe(gpt2_input, max_new_tokens=20, num_return_sequences=1, pad_token_id=50256)
                        generated_text = results[0]['generated_text']
                        st.success(f"**生成结果:**\n\n{generated_text}")


# --- Tab 4: GPT-2 Perplexity Calculation ---
with tab4:
    st.header("基于 GPT-2 的文本困惑度 (Perplexity) 计算")
    st.markdown("输入多行句子，计算每句的 Cross-Entropy Loss 及 PPL。")
    
    @st.cache_resource
    def load_gpt2_model():
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        model.eval()
        return tokenizer, model
        
    with st.spinner("正在加载 GPT-2 模型和分词器..."):
        try:
            tokenizer, gpt2_model = load_gpt2_model()
            ppl_models_loaded = True
        except Exception as e:
            st.error(f"模型加载失败: {e}")
            ppl_models_loaded = False
            
    if ppl_models_loaded:
        test_sentences_input = st.text_area(
            "输入测试句子（每行一句）", 
            "The quick brown fox jumps over the lazy dog.\nI am very happy today.\nColorless green ideas sleep furiously.\nasdf jkl qwerty uiop zxcv bnmm.", 
            height=150
        )
        
        if st.button("计算困惑度 (PPL)"):
            sentences = [s.strip() for s in test_sentences_input.split('\n') if s.strip()]
            
            if not sentences:
                st.error("请输入至少一个句子！")
            else:
                with st.spinner("正在计算 PPL..."):
                    results = []
                    
                    with torch.no_grad():
                        for sent in sentences:
                            inputs = tokenizer(sent, return_tensors="pt")
                            input_ids = inputs["input_ids"]
                            
                            if input_ids.size(1) < 2:
                                results.append({
                                    "句子": sent, 
                                    "Loss (交叉熵损失)": "N/A (太短)", 
                                    "Perplexity (困惑度)": "N/A"
                                })
                                continue
                                
                            outputs = gpt2_model(input_ids, labels=input_ids)
                            loss = outputs.loss.item()
                            ppl = np.exp(loss)
                            
                            results.append({
                                "句子": sent, 
                                "Loss (交叉熵损失)": f"{loss:.4f}", 
                                "Perplexity (困惑度)": f"{ppl:.4f}"
                            })
                            
                    df_ppl = pd.DataFrame(results)
                    st.table(df_ppl)
