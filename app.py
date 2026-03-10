import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 日本語フォントなどのワーニング対策
sns.set_theme(style="whitegrid")

# ページの基本設定
st.set_page_config(page_title="Kaggle EDA App", layout="wide")

st.title("Kaggle Data EDA App")

# サイドバーでファイル選択
st.sidebar.header("データ選択")
# submission.csvは実際のファイル名であるsample_submission.csvにマッピングします
file_options = {
    "Train Data": "train.csv",
    "Test Data": "test.csv",
    "Submission Data": "sample_submission.csv"
}
selected_option = st.sidebar.selectbox("読み込むファイルを選択してください", list(file_options.keys()))
selected_file = file_options[selected_option]

st.sidebar.markdown("---")
st.sidebar.header("パフォーマンス設定")
sample_frac = st.sidebar.slider("サンプリング割合 (%)", min_value=1, max_value=100, value=10, step=1)

# データの読み込み
@st.cache_data
def load_data(filename):
    try:
        return pd.read_csv(filename)
    except FileNotFoundError:
        return None

raw_df = load_data(selected_file)

if raw_df is not None:
    original_rows = len(raw_df)
    # サンプリングを適用
    df = raw_df.sample(frac=sample_frac / 100.0, random_state=42)
    current_rows = len(df)
    
    st.sidebar.markdown(f"**現在表示中の行数 / 元の総行数**\n\n{current_rows:,} / {original_rows:,} 行")
    st.subheader(f"{selected_option} - プレビュー (先頭10行)")
    st.dataframe(df.head(10))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("基本統計量")
        st.dataframe(df.describe())
        
    with col2:
        st.subheader("欠損値の数")
        missing_count = df.isnull().sum()
        missing_df = pd.DataFrame({
            "欠損数": missing_count,
            "欠損率(%)": (missing_count / len(df)) * 100
        })
        st.dataframe(missing_df)

    st.markdown("---")
    
    # データ型実質判定関数を追加 (閾値を10に変更)
    def is_categorical(series, max_unique=10):
        if not pd.api.types.is_numeric_dtype(series):
            return True
        return series.nunique() <= max_unique

    # タブの作成
    tab1, tab2, tab3 = st.tabs(["🎯 目的変数(Target)分析", "⚖ Train/Test分布比較", "📊 単変量解析(全特徴量)"])
    
    # === タブ1: 目的変数（Target）分析 ===
    with tab1:
        st.subheader("目的変数（Target）との関係性分析")
        target_col = st.selectbox("目的変数（Target）を選択してください", df.columns.tolist(), key="tab1_target")
        
        target_is_cat = is_categorical(df[target_col])
        
        feature_type_tab1 = st.radio("比較する特徴量のデータ型を選択", ["数値カラム", "カテゴリカラム"], key="tab1_radio")
        
        if feature_type_tab1 == "数値カラム":
            candidate_features = [c for c in df.columns if not is_categorical(df[c])]
        else:
            candidate_features = [c for c in df.columns if is_categorical(df[c])]
            
        candidate_features = [c for c in candidate_features if c != target_col]
        
        if len(candidate_features) > 0:
            if st.button("関係性を一括可視化", key="target_btn"):
                st.info(f"{target_col} と {feature_type_tab1} の全特徴量（{len(candidate_features)}個）の関係を描画します。")
                
                cols_per_row = 2
                for i in range(0, len(candidate_features), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j in range(cols_per_row):
                        idx = i + j
                        if idx < len(candidate_features):
                            feature_col = candidate_features[idx]
                            with cols[j]:
                                fig, ax = plt.subplots(figsize=(6, 4))
                                feature_is_cat = is_categorical(df[feature_col])
                                try:
                                    if not target_is_cat and not feature_is_cat:
                                        # 両方数値: 散布図
                                        sns.scatterplot(data=df, x=feature_col, y=target_col, ax=ax, alpha=0.5)
                                        ax.set_title(f"{target_col} vs {feature_col} (Scatter)")
                                    elif not target_is_cat and feature_is_cat:
                                        # Target数値, Featureカテゴリ: 箱ひげ図
                                        sns.boxplot(data=df, x=feature_col, y=target_col, ax=ax)
                                        ax.set_title(f"{target_col} by {feature_col} (Box Plot)")
                                        ax.tick_params(axis='x', rotation=45)
                                    elif target_is_cat and not feature_is_cat:
                                        # Targetカテゴリ, Feature数値: 箱ひげ図
                                        sns.boxplot(data=df, x=target_col, y=feature_col, ax=ax)
                                        ax.set_title(f"{feature_col} dist by {target_col} (Box Plot)")
                                    else:
                                        # 両方カテゴリまたは実質カテゴリ: カウントプロット (hue=target_col)
                                        # カテゴリ数が多い場合はトップ10のみ表示
                                        order = df[feature_col].value_counts().index[:10]
                                        sns.countplot(data=df, x=feature_col, hue=target_col, ax=ax, order=order)
                                        ax.set_title(f"{feature_col} counts by {target_col}")
                                        ax.tick_params(axis='x', rotation=45)
                                    st.pyplot(fig)
                                except Exception as e:
                                    st.warning(f"「{feature_col}」の描画中にエラーが発生しました: {e}")
        else:
            st.info(f"選択可能な{feature_type_tab1}がありません。")

    # === タブ2: TrainとTestの分布比較 ===
    with tab2:
        st.subheader("Train/Test データ分布の比較 (Covariate Shift)")
        raw_train_df = load_data("train.csv")
        raw_test_df = load_data("test.csv")
        
        if raw_train_df is not None and raw_test_df is not None:
            train_df = raw_train_df.sample(frac=sample_frac / 100.0, random_state=42)
            test_df = raw_test_df.sample(frac=sample_frac / 100.0, random_state=42)
            
            common_cols = list(set(train_df.columns) & set(test_df.columns))
            if len(common_cols) > 0:
                comp_type = st.radio("比較するカラムのデータ型を選択", ["数値カラム", "カテゴリカラム"], key="tab2_radio")
                
                if comp_type == "数値カラム":
                    valid_cols = [c for c in common_cols if not is_categorical(train_df[c]) and not is_categorical(test_df[c])]
                else:
                    valid_cols = [c for c in common_cols if is_categorical(train_df[c]) or is_categorical(test_df[c])]
                    
                valid_cols = sorted(valid_cols)
                
                if len(valid_cols) > 0:
                    st.info(f"共通する {comp_type}（{len(valid_cols)}個）の分布比較を一括表示します。")
                    if st.button("分布比較を一括可視化", key="compare_btn"):
                        cols_per_row = 2
                        for i in range(0, len(valid_cols), cols_per_row):
                            cols = st.columns(cols_per_row)
                            for j in range(cols_per_row):
                                idx = i + j
                                if idx < len(valid_cols):
                                    selected_compare_col = valid_cols[idx]
                                    with cols[j]:
                                        fig, ax = plt.subplots(figsize=(6, 4))
                                        
                                        if comp_type == "数値カラム":
                                            sns.histplot(data=train_df, x=selected_compare_col, color="blue", label="Train", kde=True, stat="density", common_norm=False, alpha=0.3, ax=ax)
                                            sns.histplot(data=test_df, x=selected_compare_col, color="orange", label="Test", kde=True, stat="density", common_norm=False, alpha=0.3, ax=ax)
                                            ax.set_title(f"Train vs Test Dist for {selected_compare_col}")
                                        else:
                                            # Categorical countplot stacked or grouped
                                            concat_df = pd.concat([
                                                train_df[[selected_compare_col]].assign(Dataset="Train"),
                                                test_df[[selected_compare_col]].assign(Dataset="Test")
                                            ])
                                            top_cats = concat_df[selected_compare_col].value_counts().index[:10]
                                            concat_df = concat_df[concat_df[selected_compare_col].isin(top_cats)]
                                            
                                            sns.countplot(data=concat_df, x=selected_compare_col, hue="Dataset", palette={"Train": "blue", "Test": "orange"}, alpha=0.7, ax=ax)
                                            ax.set_title(f"Train vs Test Count for {selected_compare_col}")
                                            ax.tick_params(axis='x', rotation=45)
                                        
                                        ax.legend()
                                        st.pyplot(fig)
                else:
                    st.info(f"共通する{comp_type}がありません。")
            else:
                st.warning("TrainとTestに共通するカラムがありません。")
        else:
            if train_df is None:
                st.error("同階層に `train.csv` が見つからないため、比較できません。")
            if test_df is None:
                st.error("同階層に `test.csv` が見つからないため、比較できません。")
                
    # === タブ3: 単変量解析（全特徴量） ===
    with tab3:
        st.subheader("単変量解析（全特徴量）")
        all_cols = df.columns.tolist()
        
        if len(all_cols) > 0:
            st.info(f"全 {len(all_cols)} 個の特徴量を自動でグリッド表示しています。")
            
            # 見やすくするために3列のグリッドで表示
            cols_per_row = 3
            for i in range(0, len(all_cols), cols_per_row):
                cols = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    idx = i + j
                    if idx < len(all_cols):
                        col_name = all_cols[idx]
                        with cols[j]:
                            fig, ax = plt.subplots(figsize=(5, 4))
                            if not is_categorical(df[col_name]):
                                sns.histplot(df[col_name].dropna(), kde=True, ax=ax, color="teal")
                                ax.set_title(col_name)
                            else:
                                # カテゴリが多すぎる場合を考慮し、トップ10件のみ表示
                                order = df[col_name].value_counts().index[:10]
                                sns.countplot(data=df, x=col_name, ax=ax, color="mediumpurple", order=order)
                                ax.tick_params(axis='x', rotation=45)
                                if len(df[col_name].dropna().unique()) > 10:
                                    ax.set_title(f"{col_name} (Top 10)")
                                else:
                                    ax.set_title(col_name)
                                
                            ax.set_xlabel("")
                            ax.set_ylabel("")
                            st.pyplot(fig)
        else:
            st.warning("このデータセットにはカラムがありません。")
else:
    st.error(f"{selected_file} が見つかりませんでした。Kaggleのデータセットが同じフォルダに存在するか確認してください。")
