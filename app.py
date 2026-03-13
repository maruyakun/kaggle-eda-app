import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import plotly.express as px

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

@st.cache_data
def encode_binary_features(df):
    if df is None:
        return None
    df_encoded = df.copy()
    for col in df_encoded.columns:
        # object または category 型の場合
        if df_encoded[col].dtype == 'object' or isinstance(df_encoded[col].dtype, pd.CategoricalDtype):
            unique_vals = df_encoded[col].dropna().unique()
            if len(unique_vals) == 2:
                # 文字列の小文字をキー、元の値をバリューとする辞書
                val_lower = {str(v).lower(): v for v in unique_vals}
                
                if set(val_lower.keys()) == {'yes', 'no'}:
                    mapping = {val_lower['yes']: 1, val_lower['no']: 0}
                elif set(val_lower.keys()) == {'true', 'false'}:
                    mapping = {val_lower['true']: 1, val_lower['false']: 0}
                elif set(val_lower.keys()) == {'y', 'n'}:
                    mapping = {val_lower['y']: 1, val_lower['n']: 0}
                else:
                    # それ以外の任意の2値は、文字としてソートして1, 0に割り当て
                    sorted_vals = sorted(unique_vals, key=str)
                    mapping = {sorted_vals[1]: 1, sorted_vals[0]: 0}
                
                df_encoded[col] = df_encoded[col].map(mapping)
    return df_encoded

# データロードと自動エンコーディング
raw_df = encode_binary_features(load_data(selected_file))

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
        "🎯 ターゲット分析", 
        "⚖ Train/Test分布比較",
        "🧮 動的クロス集計(ピボット)"
    ])
    tab_quality, tab_univariate, tab_corr, tab_target, tab_compare, tab_pivot = tabs
    
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
        
        col_c1, col_c2 = st.columns(2)
        # 1. 除外カラム選択機能の追加
        with col_c1:
            exclude_cols = st.multiselect(
                "除外するカラムを選択（例: ID等）", 
                df.columns.tolist(),
                key="corr_exclude_select"
            )
            
        with col_c2:
            # セレクトボックスで目的変数を指定（選択しないことも可能）
            all_cols_with_none = ["None"] + [c for c in df.columns if c not in exclude_cols]
            corr_target = st.selectbox("相関を確認したい目的変数（任意）", all_cols_with_none, key="corr_target_select")
        
        # 2. 計算対象を「数値カラム」に限定 (除外カラムを反映)
        target_df = df.drop(columns=exclude_cols)
        num_cols = target_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 相関計算用のDataFrameを作成
        corr_df = target_df[num_cols].copy()
        
        # 3. 目的変数（Target）の安全な自動エンコーディング
        if corr_target != "None" and corr_target in target_df.columns and corr_target not in num_cols:
            unique_vals = target_df[corr_target].dropna().unique()
            if len(unique_vals) == 2:
                # 2値ならpd.factorizeで安全に0/1に変換して追加
                try:
                    corr_df[corr_target], _ = pd.factorize(target_df[corr_target])
                    st.info(f"「{corr_target}」は2値のカテゴリ変数のため、相関計算用に数値（1/0）に一時変換して含めました。")
                except Exception as e:
                    st.warning(f"「{corr_target}」のエンコーディング中にエラーが発生したため、相関行列からは除外されます: {e}")
            else:
                st.warning(f"「{corr_target}」は3種類以上の値を持つカテゴリ変数のため、相関行列からは除外されます。")
        
        if len(corr_df.columns) > 1:
            fig, ax = plt.subplots(figsize=(10, max(6, len(corr_df.columns) * 0.4)))
            corr = corr_df.corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax, fmt=".2f")
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)
        else:
            st.info("相関行列を計算するための数値カラムが不足しています。")

    # === 4. ターゲット分析 ===
    with tab_target:
        st.subheader("ターゲット分析 (Target Analysis)")
        target_col = st.selectbox("目的変数（Target）を選択してください", df.columns.tolist(), key="target_col_select")
        
        target_is_cat = is_categorical(df[target_col])
        
        categorical_features = [c for c in df.columns if is_categorical(df[c]) and c != target_col]
        
        if len(categorical_features) == 0:
            st.warning("比較対象となるカテゴリ変数が存在しません。")
        else:
            feature_col = st.selectbox("比較するカテゴリ変数を選択してください", categorical_features, key="target_feature_select")
            
            clean_df = df[[target_col, feature_col]].dropna()
            
            if len(clean_df) == 0:
                st.warning("有効なデータがありません（欠損値を除外した結果0件になりました）。")
            else:
                if not target_is_cat:
                    # 数値型（回帰タスク）の場合
                    st.info(f"「{target_col}」は数値型として判定されました。カテゴリごとの平均値と分布を比較します。")
                    
                    # 平均値と標準偏差（エラーバー用）の計算
                    agg_df = clean_df.groupby(feature_col)[target_col].agg(['mean', 'std']).reset_index()
                    agg_df = agg_df.fillna(0) # stdがNaNになる場合(N=1等)の対策
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        fig_bar = px.bar(
                            agg_df, 
                            x=feature_col, 
                            y='mean', 
                            error_y='std',
                            title=f"{target_col} の平均値 (by {feature_col})",
                            labels={'mean': f'{target_col} Mean'},
                            color=feature_col
                        )
                        fig_bar.update_layout(showlegend=False)
                        st.plotly_chart(fig_bar, use_container_width=True)
                    
                    with col2:
                        fig_box = px.box(
                            clean_df, 
                            x=feature_col, 
                            y=target_col, 
                            title=f"{target_col} の分布 (by {feature_col})",
                            color=feature_col
                        )
                        fig_box.update_layout(showlegend=False)
                        st.plotly_chart(fig_box, use_container_width=True)
                        
                else:
                    # カテゴリ/二値型（分類タスク）の場合
                    st.info(f"「{target_col}」はカテゴリ/二値型として判定されました。カテゴリごとの割合を比較します。")
                    
                    # 100%積み上げ棒グラフ
                    fig_stack = px.histogram(
                        clean_df.astype(str), # 全て文字列にしてカテゴリとして扱う
                        x=feature_col, 
                        color=target_col, 
                        barnorm="percent",
                        title=f"{target_col} の割合 (by {feature_col})",
                        labels={feature_col: feature_col, 'count': '割合'}
                    )
                    fig_stack.update_layout(yaxis_title="割合 (100%)", margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_stack, use_container_width=True)

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
                                            train_data = train_df[[selected_compare_col]].dropna()
                                            test_data = test_df[[selected_compare_col]].dropna()
                                            
                                            if len(train_data) > 1 and len(test_data) > 1:
                                                # px.histogramを使ってヒストグラムのみを描画（KDEやfigure_factoryを回避し軽量化）
                                                plot_df = pd.concat([
                                                    train_data.assign(Dataset="Train"),
                                                    test_data.assign(Dataset="Test")
                                                ])
                                                
                                                fig_plotly = px.histogram(
                                                    plot_df,
                                                    x=selected_compare_col,
                                                    color="Dataset",
                                                    barmode="overlay",
                                                    histnorm="probability",
                                                    opacity=0.6,
                                                    title=f"Train vs Test Dist for {selected_compare_col}",
                                                    color_discrete_map={"Train": "#1f77b4", "Test": "#ff7f0e"}
                                                )
                                                fig_plotly.update_layout(
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

    # === 6. 動的クロス集計（ピボット） ===
    with tab_pivot:
        st.subheader("動的クロス集計 (Dynamic Pivot Table)")
        
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        
        all_cols_options = ["None"] + df.columns.tolist()
        with col_p1:
            pivot_index = st.selectbox("行 (Index)", all_cols_options, index=1 if len(all_cols_options) > 1 else 0)
        with col_p2:
            pivot_columns = st.selectbox("列 (Columns)", all_cols_options, index=0)
        with col_p3:
            pivot_values = st.selectbox("集計対象の値 (Values)", all_cols_options, index=0)
        with col_p4:
            pivot_aggfunc = st.selectbox("集計関数", ["count", "mean", "sum", "min", "max"], index=0)
            
        if pivot_index != "None":
            try:
                # pivot_tableの実行
                if pivot_aggfunc == "count":
                    if pivot_columns != "None":
                        pivot_df = pd.crosstab(df[pivot_index], df[pivot_columns])
                    else:
                        pivot_df = df[pivot_index].value_counts().to_frame("count")
                else:
                    if pivot_values == "None":
                        st.warning("mean, sum等の集計関数を使用する場合は「集計対象の値 (Values)」を選択してください。")
                        pivot_df = pd.DataFrame()
                    else:
                        pivot_col_arg = pivot_columns if pivot_columns != "None" else None
                        pivot_df = pd.pivot_table(df, index=pivot_index, columns=pivot_col_arg, values=pivot_values, aggfunc=pivot_aggfunc)
                
                if not pivot_df.empty:
                    # 数値型に背景グラデーションを適用して表示
                    styled_df = pivot_df.style.background_gradient(cmap="Blues", axis=None)
                    st.dataframe(styled_df, use_container_width=True)
                    
            except Exception as e:
                st.error(f"クロス集計の計算中にエラーが発生しました: {e}")
        else:
            st.info("少なくとも「行 (Index)」を選択してください。")
                
else:
    st.error(f"{selected_file} の読み込みに失敗しました。データが存在するか、URLが正しいか確認してください。")
