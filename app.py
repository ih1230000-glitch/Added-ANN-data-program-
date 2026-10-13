import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

st.set_page_config(
    page_title="인공신경망(ANN) 데이터 예측 플랫폼",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 인공신경망(ANN) 기반 데이터 예측 및 분석 플랫폼")
st.markdown("""
이 웹 애플리케이션은 **인공신경망(Artificial Neural Network, ANN)**의 핵심 학습 원리인 **다층 구조(Input-Hidden-Output), 활성화 함수(Activation Function), 가중치/편향 최적화(Backpropagation & Gradient Descent)**를 활용하여 데이터를 학습하고 예측 성능을 검증하는 전문 분석 도구입니다.
""")

st.sidebar.header("🛠️ 1. 데이터 파일 관리")
st.sidebar.markdown("기본 제공되는 **샘플 데이터**로 즉시 분석 화면을 확인할 수 있으며, 본인의 **CSV 파일**을 업로드하여 교체할 수도 있습니다.")
uploaded_file = st.sidebar.file_uploader("커스텀 CSV 파일 업로드 (선택사항)", type=["csv"])

# 파일이 없으면 기본 샘플 데이터 자동 로드 (허전함 방지)
if uploaded_file is None:
    st.sidebar.info("💡 현재 **기본 샘플 데이터**가 적용되어 있습니다. 본인 파일로 분석하려면 위에서 업로드하세요!")
    sample_data = """X,Y
1.0,2.1
2.0,3.8
3.0,7.9
4.0,15.2
5.0,24.8
6.0,36.1
7.0,49.5
8.0,64.2
9.0,81.0
10.0,100.5
11.0,121.2
12.0,143.8
13.0,168.1
14.0,196.0
15.0,225.2
16.0,256.4
17.0,289.1
18.0,324.5
19.0,361.0
20.0,400.2"""
    df = pd.read_csv(io.StringIO(sample_data))
    data_source_name = "기본 샘플 데이터 (Quadratic Dataset)"
else:
    df = pd.read_csv(uploaded_file)
    data_source_name = f"사용자 업로드 파일 ({uploaded_file.name})"

with st.expander(f"📋 데이터셋 정보 [{data_source_name}]", expanded=True):
    col_info1, col_info2 = st.columns([3, 1])
    with col_info1:
        st.dataframe(df.head(), use_container_width=True)
    with col_info2:
        st.metric("총 데이터 행 수", f"{df.shape[0]}개")
        st.metric("특성(열) 수", f"{df.shape[1]}개")

columns = df.columns.tolist()

st.sidebar.header("📊 2. 변수(Feature & Target) 선택")
x_cols = st.sidebar.multiselect("X축 입력 변수 (Input Features, 다중 선택 가능)", columns, default=[columns[0]] if len(columns)>0 else [])
y_col = st.sidebar.selectbox("Y축 출력 변수 (Target)", columns, index=len(columns)-1 if len(columns)>1 else 0)

st.sidebar.header("⚙️ 3. 인공신경망 하이퍼파라미터 설정")
activation_name = st.sidebar.selectbox(
    "활성화 함수 (Activation Function)", 
    ["Sigmoid", "ReLU", "Tanh"],
    help="은닉층에서 비선형성을 부여하여 복잡한 패턴을 학습할 수 있게 만드는 함수입니다."
)
hidden_units = st.sidebar.slider("은닉층 뉴런 수 (Hidden Neurons)", 1, 32, 8, step=1)
test_size_ratio = st.sidebar.slider("테스트 데이터 분할 비율 (Test Split)", 0.1, 0.4, 0.2, step=0.05)
learning_rate = st.sidebar.number_input("학습률 (Learning Rate)", value=0.01, format="%.4f")
epochs = st.sidebar.slider("학습 반복 횟수 (Epochs)", 100, 10000, 2000, step=100)

if len(x_cols) > 0 and y_col:
    sub_cols = x_cols + [y_col]
    temp_df = df[sub_cols].copy()
    for col in sub_cols:
        temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')
    temp_df = temp_df.dropna()
    
    if len(temp_df) > 5:
        X_raw = temp_df[x_cols].values
        y_raw = temp_df[y_col].values.reshape(-1, 1)
        
        # Train / Test Split
        np.random.seed(42)
        indices = np.random.permutation(len(X_raw))
        test_cnt = int(len(X_raw) * test_size_ratio)
        test_idx, train_idx = indices[:test_cnt], indices[test_cnt:]
        
        X_train_raw, X_test_raw = X_raw[train_idx], X_raw[test_idx]
        y_train_raw, y_test_raw = y_raw[train_idx], y_raw[test_idx]
        
        # Standardization
        X_mean, X_std = X_train_raw.mean(axis=0), X_train_raw.std(axis=0)
        X_std[X_std == 0] = 1.0
        X_train = (X_train_raw - X_mean) / X_std
        X_test = (X_test_raw - X_mean) / X_std
        
        y_mean, y_std = y_train_raw.mean(), y_train_raw.std()
        if y_std == 0: y_std = 1.0
        y_train = (y_train_raw - y_mean) / y_std
        y_test = (y_test_raw - y_mean) / y_std
        
        N_train, D = X_train.shape
        H = hidden_units
        
        # Weight Initialization
        np.random.seed(42)
        W1 = np.random.randn(D, H) * np.sqrt(2.0 / D)
        b1 = np.zeros((1, H))
        W2 = np.random.randn(H, 1) * np.sqrt(2.0 / H)
        b2 = np.zeros((1, 1))
        
        def activate(z, name):
            if name == "Sigmoid":
                return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
            elif name == "ReLU":
                return np.maximum(0, z)
            elif name == "Tanh":
                return np.tanh(z)
                
        def activate_derivative(a, name):
            if name == "Sigmoid":
                return a * (1 - a)
            elif name == "ReLU":
                return (a > 0).astype(float)
            elif name == "Tanh":
                return 1 - a**2
        
        train_loss_history = []
        test_loss_history = []
        
        # Training Loop
        for epoch in range(epochs):
            Z1 = np.dot(X_train, W1) + b1
            A1 = activate(Z1, activation_name)
            Z2 = np.dot(A1, W2) + b2
            Y_pred_train = Z2
            
            train_error = Y_pred_train - y_train
            train_loss = np.mean(train_error ** 2)
            train_loss_history.append(train_loss)
            
            Z1_test = np.dot(X_test, W1) + b1
            A1_test = activate(Z1_test, activation_name)
            Y_pred_test = np.dot(A1_test, W2) + b2
            test_loss = np.mean((Y_pred_test - y_test) ** 2)
            test_loss_history.append(test_loss)
            
            dY = 2 * train_error / N_train
            dW2 = np.dot(A1.T, dY)
            db2 = np.sum(dY, axis=0, keepdims=True)
            
            dA1 = np.dot(dY, W2.T)
            dZ1 = dA1 * activate_derivative(A1, activation_name)
            dW1 = np.dot(X_train.T, dZ1)
            db1 = np.sum(dZ1, axis=0, keepdims=True)
            
            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1
            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2
        
        # Test Evaluation
        Z1_final = np.dot(X_test, W1) + b1
        A1_final = activate(Z1_final, activation_name)
        y_pred_test_scaled = np.dot(A1_final, W2) + b2
        y_pred_test_orig = y_pred_test_scaled * y_std + y_mean
        
        mse = np.mean((y_test_raw - y_pred_test_orig) ** 2)
        mae = np.mean(np.abs(y_test_raw - y_pred_test_orig))
        ss_tot = np.sum((y_test_raw - y_test_raw.mean()) ** 2)
        ss_res = np.sum((y_test_raw - y_pred_test_orig) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
        st.markdown("---")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📈 인공신경망 학습 및 성능 시각화")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
            
            if D == 1:
                sort_idx = np.argsort(X_test_raw[:, 0])
                ax1.scatter(X_train_raw[:, 0], y_train_raw, color='dodgerblue', alpha=0.6, label='학습 데이터 (Train)', s=30)
                ax1.scatter(X_test_raw[:, 0], y_test_raw, color='crimson', alpha=0.8, label='테스트 실제값 (Test)', s=40, zorder=5)
                ax1.plot(X_test_raw[sort_idx, 0], y_pred_test_orig[sort_idx], color='rebeccapurple', linewidth=2.5, label='ANN 예측선', zorder=6)
                ax1.set_xlabel(x_cols[0])
                ax1.set_ylabel(y_col)
            else:
                ax1.scatter(y_test_raw, y_pred_test_orig, color='teal', alpha=0.8, s=40)
                ax1.plot([y_test_raw.min(), y_test_raw.max()], [y_test_raw.min(), y_test_raw.max()], 'r--', label='이상적 예측선')
                ax1.set_xlabel("테스트 실제값 (Actual)")
                ax1.set_ylabel("테스트 예측값 (Predicted)")
                
            ax1.set_title("실제값 vs 신경망 예측")
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.5)
            
            ax2.plot(train_loss_history, color='royalblue', label='Train Loss', linewidth=1.5)
            ax2.plot(test_loss_history, color='darkorange', label='Test Loss', linewidth=1.5, linestyle='--')
            ax2.set_title("에폭별 손실 함수 (MSE Loss)")
            ax2.set_xlabel("Epochs")
            ax2.set_ylabel("Loss")
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.5)
            
            st.pyplot(fig)
        
        with col2:
            st.subheader("📌 정량적 성능 평가 및 요약")
            st.success(f"**네트워크 구조:**\n- 입력층: {D}개 변수\n- 은닉층: {H}개 뉴런 ({activation_name})\n- 출력층: 1개")
            st.metric(label="테스트 MSE", value=f"{mse:.4f}")
            st.metric(label="테스트 MAE", value=f"{mae:.4f}")
            st.metric(label="결정 계수 ($R^2$ Score)", value=f"{r2:.4f}")
        
        # Single instance prediction simulator
        st.markdown("---")
        st.subheader("🔮 실시간 단건 예측 시뮬레이터")
        st.markdown("입력 변수($X$) 값을 직접 조절하여 학습된 인공신경망의 예측 결과($Y$)를 즉시 확인해보세요!")
        
        user_inputs = []
        pred_cols = st.columns(min(len(x_cols), 4))
        for i, col_name in enumerate(x_cols):
            col_min = float(temp_df[col_name].min())
            col_max = float(temp_df[col_name].max())
            col_mean = float(temp_df[col_name].mean())
            with pred_cols[i % 4]:
                val = st.number_input(f"{col_name}", value=col_mean, min_value=col_min-(col_max-col_min)*0.2, max_value=col_max+(col_max-col_min)*0.2)
                user_inputs.append(val)
        
        if st.button("🚀 인공신경망 예측 수행하기", use_container_width=True):
            single_x_raw = np.array(user_inputs).reshape(1, -1)
            single_x_scaled = (single_x_raw - X_mean) / X_std
            
            s_z1 = np.dot(single_x_scaled, W1) + b1
            s_a1 = activate(s_z1, activation_name)
            s_z2 = np.dot(s_a1, W2) + b2
            single_y_pred = s_z2 * y_std + y_mean
            
            st.info(f"✨ **인공신경망 예측 결과 ({y_col}):** **{single_y_pred[0, 0]:.4f}**")

    else:
        st.warning("⚠️ 유효한 숫자 데이터가 5개 이상 필요합니다.")
else:
    st.warning("⚠️ X축 입력 변수와 Y축 출력 변수를 각각 선택해 주세요.")
