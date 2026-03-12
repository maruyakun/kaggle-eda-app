import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff

# 日本語フォントなどのワーニング対策
sns.set_theme(style="whitegrid")

# ページの基本設定
st.set_page_config(page_title="Kaggle EDA App", layout="wide")

st.title("Kaggle Data EDA App")

# サイドバーでファイル選択
st.sidebar.header("データ選択")
# submission.csvは実際のファイル名であるsample_submission.csvにマッピングします
file_options = {
    "Train Data": "https://uholddjjjwpuvmvrzylz.supabase.co/storage/v1/object/public/kaggle-data/train.parquet",
    "Test Data": "https://uholddjjjwpuvmvrzylz.supabase.co/storage/v1/object/public/kaggle-data/test.parquet",
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
        if str(filename).endswith(".parquet") or ".parquet" in str(filename):
            return pd.read_parquet(filename)
        else:
            return pd.read_csv(filename)
    except Exception:
        return None

raw_df = load_data(selected_file)

if raw_df is not None:
    original_rows = len(raw_df)
    # サンプリングを適用
    df = raw_df.sample(frac=sample_frac / 100.0, random_state=42)
    current_rows = len(df)
    
    st.sidebar.markdown(f"**現在表示中の行数 / 元の総行数**\n\n{current_rows:,} / {original_rows:,} 行")
    
    # === 1. アラートの自動表示 ===
    high_missing_cols = [col for col in df.columns if df[col].isnull().mean() >= 0.5]
    zero_var_cols = [col for col in df.columns if df[col].nunique() == 1]
    
    if high_missing_cols or zero_var_cols:
        st.subheader("データ品質アラート")
        if high_missing_cols:
            st.warning(f"欠損値が50%以上含まれるカラム: {', '.join(high_missing_cols)}")
        if zero_var_cols:
            st.info(f"値が1種類しか存在しない（分散がゼロの）カラム: {', '.join(zero_var_cols)}")
            
    st.subheader(f"{selected_option} - プレビュー (先頭10行)")
    st.dataframe(df.head(10))
    
    st.markdown("---")
    
    # データ型実質判定関数を追加 (閾値を10に変更)
    def is_categorical(series, max_unique=10):
        if not pd.api.types.is_numeric_dtype(series):
            return True
        return series.nunique() <= max_unique

    # === 2. タブの順番変更 (指定の論理順に再配置) ===
    tabs = st.tabs([
        "📊 データ品質と基礎統計", 
        "📈 単変量解析(全特徴量)",
        "🔗 相関分析", 
        "🎯 目的変数(Target)分析", 
        "⚖ Train/Test分布比較"
    ])
    tab_quality, tab_univariate, tab_corr, tab_target, tab_compare = tabs
    
    # === 1. データ品質と基礎統計 ===
    with tab_quality:
        st.subheader("データ品質と基礎統計 (Data Quality & Stats)")
        
        col_q1, col_q2 = st.columns(2)
        with col_q1:
            st.write("#### 数値カラムの基礎統計量")
            st.dataframe(df.describe())
            
        with col_q2:
            st.write("#### 全カラムの欠損値割合")
            missing_rate = (df.isnull().sum() / len(df)) * 100
            missing_rate = missing_rate[missing_rate > 0].sort_values(ascending=False)
            
            if not missing_rate.empty:
                fig, ax = plt.subplots(figsize=(6, max(4, len(missing_rate) * 0.4)))
                sns.barplot(x=missing_rate.values, y=missing_rate.index, ax=ax, palette="viridis")
                ax.set_xlabel("Missing Rate (%)")
                ax.set_ylabel("Features")
                ax.set_title("Missing Rate per Feature")
                st.pyplot(fig)
            else:
                st.success("欠損値を含むカラムはありません。")

    # === 2. 単変量解析（全特徴量） ===
    with tab_univariate:
        st.subheader("単変量解析（全特徴量）")
        all_cols = df.columns.tolist()
        
        if len(all_cols) > 0:
            st.info(f"全 {len(all_cols)} 個の特徴量を自動でグリッド表示しています。")
            
            # 一括展開トグルを追加
            expand_all_uni = st.toggle("全てのグラフを展開して表示", value=False, key="toggle_uni")
            
            # 見やすくするために3列のグリッドで表示
            cols_per_row = 3
            for i in range(0, len(all_cols), cols_per_row):
                cols = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    idx = i + j
                    if idx < len(all_cols):
                        col_name = all_cols[idx]
                        with cols[j]:
                            # トグルの状態を expanded 引数に連動
                            with st.expander(f"{col_name} の分布", expanded=expand_all_uni):
                                fig, ax = plt.subplots(figsize=(5, 4))
                                if not is_categorical(df[col_name]):
                                    sns.histplot(df[col_name].dropna(), kde=True, ax=ax, color="teal")
                                    ax.set_title(col_name)
                                else:
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

    # === 3. 相関分析 ===
    with tab_corr:
        st.subheader("相関分析 (Correlation)")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr = df[num_cols].corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax, fmt=".2f")
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)
        else:
            st.info("相関行列を計算するための数値カラムが不足しています。")

    # === 4. 目的変数（Target）分析 ===
    with tab_target:
        st.subheader("目的変数（Target）との関係性分析")
        target_col = st.selectbox("目的変数（Target）を選択してください", df.columns.tolist(), key="tab1_target")
        
        target_is_cat = is_categorical(df[target_col])
        
        feature_type_tab1 = st.radio("比較する特徴量のデータ型を選択", ["数値カラム", "カテゴリカラム"], key="tab1_radio")
        
        if feature_type_tab1 == "数値カラム":
            candidate_features = [c for c in df.columns if not is_categorical(df[c])]
        else:
            candidate_features = [c for c in df.columns if is_categorical(df[c])]
            
        candidate_features = [c for c in candidate_features if c != target_col]
        
        # 目的変数が数値で比較対象も数値の場合、相関係数（絶対値）で降順ソート
        if not target_is_cat and feature_type_tab1 == "数値カラム" and len(candidate_features) > 0:
            corrs = df[[target_col] + candidate_features].corr()[target_col].drop(target_col).abs()
            candidate_features = corrs.sort_values(ascending=False).index.tolist()
            
        if len(candidate_features) > 0:
            st.info(f"{target_col} と {feature_type_tab1} の全特徴量（{len(candidate_features)}個）の関係を描画します。")
            
            # 一括展開トグルを追加
            expand_all_target = st.toggle("全てのグラフを展開して表示", value=False, key="toggle_target")
            
            cols_per_row = 2
            for i in range(0, len(candidate_features), cols_per_row):
                cols = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    idx = i + j
                    if idx < len(candidate_features):
                        feature_col = candidate_features[idx]
                        with cols[j]:
                            # トグルの状態を expanded 引数に連動
                            with st.expander(f"{feature_col} の分布", expanded=expand_all_target):
                                fig, ax = plt.subplots(figsize=(6, 4))
                                feature_is_cat = is_categorical(df[feature_col])
                                try:
                                    if not target_is_cat and not feature_is_cat:
                                        sns.scatterplot(data=df, x=feature_col, y=target_col, ax=ax, alpha=0.5)
                                        ax.set_title(f"{target_col} vs {feature_col} (Scatter)")
                                    elif not target_is_cat and feature_is_cat:
                                        sns.boxplot(data=df, x=feature_col, y=target_col, ax=ax)
                                        ax.set_title(f"{target_col} by {feature_col} (Box Plot)")
                                        ax.tick_params(axis='x', rotation=45)
                                    elif target_is_cat and not feature_is_cat:
                                        sns.boxplot(data=df, x=target_col, y=feature_col, ax=ax)
                                        ax.set_title(f"{feature_col} dist by {target_col} (Box Plot)")
                                    else:
                                        order = df[feature_col].value_counts().index[:10]
                                        sns.countplot(data=df, x=feature_col, hue=target_col, ax=ax, order=order)
                                        ax.set_title(f"{feature_col} counts by {target_col}")
                                        ax.tick_params(axis='x', rotation=45)
                                    st.pyplot(fig)
                                except Exception as e:
                                    st.warning(f"「{feature_col}」の描画中にエラーが発生しました: {e}")
        else:
            st.info(f"選択可能な{feature_type_tab1}がありません。")

    # === 5. TrainとTestの分布比較 ===
    with tab_compare:
        st.subheader("Train/Test データ分布の比較 (Covariate Shift)")
        raw_train_df = load_data(file_options["Train Data"])
        raw_test_df = load_data(file_options["Test Data"])
        
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
                    
                    # 一括展開トグルを追加
                    expand_all_compare = st.toggle("全てのグラフを展開して表示", value=False, key="toggle_compare")
                    
                    cols_per_row = 2
                    for i in range(0, len(valid_cols), cols_per_row):
                        cols = st.columns(cols_per_row)
                        for j in range(cols_per_row):
                            idx = i + j
                            if idx < len(valid_cols):
                                selected_compare_col = valid_cols[idx]
                                with cols[j]:
                                    # トグルの状態を expanded 引数に連動
                                    with st.expander(f"{selected_compare_col} の分布", expanded=expand_all_compare):
                                        if comp_type == "数値カラム":
                                            # エラー防止のため、欠損値（NaN）を事前に除外
                                            train_data = train_df[selected_compare_col].dropna()
                                            test_data = test_df[selected_compare_col].dropna()
                                            
                                            if len(train_data) > 1 and len(test_data) > 1:
                                                # Plotlyのfigure_factoryを使ってKDEとヒストグラム（確率密度）を描画
                                                fig_plotly = ff.create_distplot(
                                                    [train_data, test_data],
                                                    group_labels=['Train', 'Test'],
                                                    show_hist=True,
                                                    show_rug=False,
                                                    colors=['#1f77b4', '#ff7f0e']
                                                )
                                                fig_plotly.update_layout(
                                                    title_text=f"Train vs Test Dist for {selected_compare_col}",
                                                    barmode='overlay',
                                                    margin=dict(l=20, r=20, t=40, b=20)
                                                )
                                                st.plotly_chart(fig_plotly, use_container_width=True)
                                            else:
                                                st.warning(f"「{selected_compare_col}」には分布を描画するのに十分なデータがありません。")
                                        else:
                                            # カテゴリカラムの場合は従来のSeaborn描画を維持
                                            fig, ax = plt.subplots(figsize=(6, 4))
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
                st.error("Trainデータの読み込みに失敗したため、比較できません。")
            if test_df is None:
                st.error("Testデータの読み込みに失敗したため、比較できません。")
                
else:
    st.error(f"{selected_file} の読み込みに失敗しました。データが存在するか、URLが正しいか確認してください。")
