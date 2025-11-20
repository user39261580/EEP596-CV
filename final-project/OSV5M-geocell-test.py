import pandas as pd
import numpy as np
from sklearn.cluster import OPTICS
from sklearn.neighbors import BallTree
from tqdm import tqdm
import warnings

# 忽略一些 Pandas 的 SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

def haversine_dist(lat1, lon1, lat2, lon2):
    """計算兩點間的 Haversine 距離 (回傳單位: 公里)"""
    R = 6371.0  # 地球半徑
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def create_pigeon_geocells(df, min_samples=200, optics_min_samples=30):
    """
    實作 Stanford PIGEON 的 GeoCell 設計方法
    
    Args:
        df: OSV5M 資料集的 DataFrame，必須包含 latitude, longitude, unique_city, unique_region, country
        min_samples: PIGEON 論文中的 MINSIZE (每個 cell 最小樣本數)
        optics_min_samples: OPTICS 分群的最小樣本參數
    
    Returns:
        df: 包含新欄位 'geocell_id' 的 DataFrame
        cell_metadata: 每個 geocell 的資訊 (centroid, size)
    """
    print(f"=== 開始 PIGEON GeoCell 生成流程 (Total samples: {len(df)}) ===")
    
    # 1. 初始化：建立基礎行政區單元 (Semantic Initialization)
    # PIGEON 使用 Admin 2 (類似 City/District) 作為基礎
    # 優先使用 unique_city，若無則退回到 unique_region，再無則 country
    print("--- Step 1: 初始化基礎行政區單元 ---")
    df['base_admin_id'] = df['unique_city'].fillna(df['unique_region']).fillna(df['country'])
    
    # 簡單編碼轉成整數 ID 以便處理
    df['temp_cell_id'] = df.groupby('base_admin_id').ngroup()
    
    # 計算每個 cell 的 centroid 和 count
    def get_cell_stats(sub_df, cell_col='temp_cell_id'):
        stats = sub_df.groupby(cell_col).agg({
            'latitude': 'mean',
            'longitude': 'mean',
            'country': 'first', # 用來確保不跨國合併
            'id': 'count'
        }).rename(columns={'id': 'count', 'latitude': 'lat', 'longitude': 'lon'})
        return stats

    cell_stats = get_cell_stats(df)
    print(f"初始行政區單元數量: {len(cell_stats)}")

    # 2. 合併過小的 Cells (Semantic Merging)
    # 規則：不跨越 Country，盡量在 Region 內，尋找最近鄰居合併
    print(f"--- Step 2: 合併樣本數 < {min_samples} 的單元 ---")
    
    # 將 dataframe 轉為 dict 以便快速更新映射
    # mapping: old_cell_id -> new_cell_id
    # 初始 mapping 是自己對應自己
    cell_mapping = {cid: cid for cid in cell_stats.index}
    
    # 為了加速搜尋，我們按國家分組處理
    countries = df['country'].unique()
    
    processed_count = 0
    
    for country in tqdm(countries, desc="Merging small cells by country"):
        # 取得該國家的所有 cells
        country_cells = cell_stats[cell_stats['country'] == country].copy()
        
        if len(country_cells) == 0:
            continue
            
        # 只要還有小於門檻的 cell 且該國還有 >1 個 cell，就繼續合併
        # 這裡使用簡化的貪婪演算法：從小到大處理
        
        # 為了效率，這裡做一次性合併：
        # 找出所有需要被合併的小 cell
        small_cells = country_cells[country_cells['count'] < min_samples]
        
        if len(small_cells) == 0:
            continue
            
        # 建立該國所有 cell 的座標樹 (用於找最近鄰居)
        # 轉換 lat/lon 到 radians
        coords_rad = np.deg2rad(country_cells[['lat', 'lon']].values)
        tree = BallTree(coords_rad, metric='haversine')
        
        for small_cid in small_cells.index:
            # 如果這個 cell 已經被併掉了 (在這一輪迴圈中)，跳過
            # 但因為我們是靜態 list，需要檢查 cell_mapping
            current_target = cell_mapping[small_cid]
            if current_target != small_cid:
                continue # 已經被合併過
            
            # 查詢最近的 k 個鄰居 (k=2, 第一個是自己)
            # Query returns distance and index in country_cells
            dist, ind = tree.query(np.deg2rad([[country_cells.loc[small_cid].lat, country_cells.loc[small_cid].lon]]), k=5)
            
            # 尋找最佳合併對象
            merged = False
            for i in range(1, 5): # 嘗試最近的幾個鄰居
                if i >= len(ind[0]): break
                
                neighbor_idx = ind[0][i]
                neighbor_cid = country_cells.index[neighbor_idx]
                
                # 取得 neighbor 目前真正的 ID (因為可能發生鍊式合併 A->B, B->C)
                # 簡化處理：只看原始鄰居，將小 cell 標記指向鄰居
                
                # 更新 mapping
                cell_mapping[small_cid] = neighbor_cid
                
                # 更新被合併後的統計資料 (簡單累加 count，座標暫不重算以節省開銷)
                # 在實作上，通常將小 cell 併入大 cell
                country_cells.at[neighbor_cid, 'count'] += country_cells.at[small_cid, 'count']
                merged = True
                break
    
    # 應用合併結果到 DataFrame
    df['geocell_id'] = df['temp_cell_id'].map(cell_mapping)
    
    # 重新整理 ID (因為中間有些 ID 消失了)
    df['geocell_id'] = df.groupby('geocell_id').ngroup()
    
    # 重新計算合併後的統計
    final_cell_stats = get_cell_stats(df, 'geocell_id')
    print(f"合併後 Cell 數量: {len(final_cell_stats)}")
    
    # 3. 分割過大的 Cells (Density Splitting with OPTICS)
    # 針對 count > 2 * min_samples 的 cell 進行 OPTICS 分群
    print(f"--- Step 3: 使用 OPTICS 分割過大單元 ( > {2*min_samples} 樣本) ---")
    
    # 找出需要分割的 candidates
    large_cells = final_cell_stats[final_cell_stats['count'] > 2 * min_samples].index
    
    next_new_id = df['geocell_id'].max() + 1
    split_count = 0
    
    for cid in tqdm(large_cells, desc="Splitting large cells"):
        # 取出該 cell 的所有資料點
        cell_mask = df['geocell_id'] == cid
        points = df.loc[cell_mask, ['latitude', 'longitude']]
        
        if len(points) > 10000:
            # 若點太多 OPTICS 會很慢，這裡可以隨機採樣或用 MiniBatchKMeans 加速
            # 為了 PIGEON 原味，我們還是跑 OPTICS 但限制參數
            pass
            
        # 轉換成 radians
        X = np.radians(points.values)
        
        # 執行 OPTICS
        # max_eps: 鄰域半徑，設為小範圍例如 5km (約 0.0008 radians) 視需求調整
        # PIGEON 論文未明講 max_eps，但通常是城市尺度
        optics = OPTICS(min_samples=optics_min_samples, metric='haversine', n_jobs=-1)
        try:
            optics.fit(X)
        except Exception as e:
            continue
            
        labels = optics.labels_
        
        # 檢查分群結果
        unique_labels = set(labels)
        if len(unique_labels) <= 1:
            continue # 無法分群 (只有 noise -1 或單一群)
            
        # 找出最大的 Cluster
        # 忽略 noise (-1)
        best_cluster_label = -1
        max_size = 0
        
        counts = pd.Series(labels).value_counts()
        for label, count in counts.items():
            if label == -1: continue
            if count > max_size:
                max_size = count
                best_cluster_label = label
        
        if best_cluster_label == -1:
            continue
            
        # 檢查分割條件：Cluster 大小 > MIN 且 剩餘部分 > MIN
        cluster_size = max_size
        remainder_size = len(points) - cluster_size
        
        if cluster_size >= min_samples and remainder_size >= min_samples:
            # 執行分割：將 Cluster 內的點賦予新的 ID
            # PIGEON 提到 Voronoi，但在標註資料集時，直接將分群結果作為 Label 即可
            # (因為那些點在空間上就是聚在一起的)
            
            # 取得屬於該 cluster 的 row indices
            cluster_indices = points.index[labels == best_cluster_label]
            
            # 更新 DataFrame
            df.loc[cluster_indices, 'geocell_id'] = next_new_id
            next_new_id += 1
            split_count += 1
            
            # PIGEON 論文提到這是 iterative 的，但為了效率通常做一輪或對超大 cell 遞迴
            # 這裡實作單層分割
            
    print(f"完成分割，共產生 {split_count} 個新 Cells")
    print(f"最終總 Cell 數量: {df['geocell_id'].nunique()}")
    
    # 4. 產生最終輸出與 Metadata
    final_stats = df.groupby('geocell_id').agg({
        'latitude': 'mean',
        'longitude': 'mean',
        'id': 'count'
    }).rename(columns={'latitude': 'centroid_lat', 'longitude': 'centroid_lon', 'id': 'num_samples'})
    
    return df[['id', 'geocell_id']], final_stats

# ==========================================
# 使用範例
# ==========================================
if __name__ == "__main__":
    # 假設你已經讀取了 CSV
    # df = pd.read_csv("osv5m_train.csv")
    
    # 這裡建立一個假的測試資料框來示範
    print("Generating dummy data for demonstration...")
    data = {
        'id': range(1000),
        'latitude': np.random.uniform(20, 50, 1000),
        'longitude': np.random.uniform(0, 20, 1000),
        'unique_city': np.random.choice(['CityA', 'CityB', 'CityC', None], 1000),
        'unique_region': np.random.choice(['Region1', 'Region2'], 1000),
        'country': ['CountryX'] * 1000
    }
    dummy_df = pd.DataFrame(data)
    
    # 執行
    labeled_df, metadata = create_pigeon_geocells(dummy_df, min_samples=10, optics_min_samples=5)
    
    print("\nResult Preview:")
    print(labeled_df.head())
    print("\nMetadata Preview:")
    print(metadata.head())
    
    # 儲存結果
    # labeled_df.to_csv("osv5m_pigeon_labels.csv", index=False)
    # metadata.to_csv("osv5m_pigeon_centroids.csv")
