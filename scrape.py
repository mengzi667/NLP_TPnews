import requests
import pandas as pd
import time
import random
from tqdm import tqdm
from bs4 import BeautifulSoup
import json

# --- 1. 配置区域 (请在此处修改你的个人信息) ---

# TODO: 替换成你从浏览器F12开发者工具中找到的、返回帖子列表JSON数据的那个真实URL
API_URL = "https://m.weibo.cn/api/container/getIndex?containerid=100103type%3D1%26q%3D%E4%B8%8A%E6%B5%B7%E5%9C%B0%E9%93%81&page_type=searchall" # 这通常是移动端的API地址，可能更稳定

# TODO: 替换成你自己的Cookie，这是成功采集的关键！请确保它是最新的。
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36',
    'Cookie': '_T_WM=73681639988; WEIBOCN_FROM=1110006030; MLOGIN=0; BAIDU_SSP_lcr=https://www.google.com/; M_WEIBOCN_PARAMS=luicode%3D10000011%26lfid%3D100103type%253D1%2526q%253D%25E4%25B8%258A%25E6%25B5%25B7%25E5%259C%25B0%25E9%2593%2581%26fid%3D100103type%253D1%2526q%253D%25E4%25B8%258A%25E6%25B5%25B7%25E5%259C%25B0%25E9%2593%2581%26uicode%3D10000011; XSRF-TOKEN=26dd72; mweibo_short_token=efa2301289',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7',
    'Connection': 'keep-alive',
    'MWeibo-Pwa': '1',
    'X-Requested-With': 'XMLHttpRequest',
    'X-XSRF-TOKEN': '26dd72', # 注意：有时候需要把Cookie里的XSRF-TOKEN单独拿出来放到这里
}

# --- 2. 主采集逻辑 ---

def scrape_weibo_data(keyword, pages_to_scrape=10):
    """
    根据关键词和页数，爬取微博数据。
    """
    scraped_data = []
    
    for page in tqdm(range(1, pages_to_scrape + 1), desc=f"正在采集'{keyword}'"):
        
        # a. 设置请求参数，这是微博搜索的常用containerid
        # containerid和q是核心参数
        params = {
            'containerid': f'100103type=1&q={keyword}',
            'page_type': 'searchall',
            'page': page
        }
        
        # b. 发送网络请求
        try:
            response = requests.get(API_URL, headers=HEADERS, params=params, timeout=20)
            response.raise_for_status() 
        except requests.exceptions.RequestException as e:
            print(f"\n在请求第 {page} 页时发生网络错误: {e}")
            break

        # c. 解析JSON数据
        try:
            data = response.json()
            
            # d. **【核心修改】** 根据你提供的JSON结构，定位到正确的卡片列表
            card_list = data.get('data', {}).get('cards', [])
            if not card_list:
                print(f"\n在第 {page} 页未找到'cards'列表，可能已到达末页。")
                break

            # e. **【核心修改】** 遍历这个复杂的、可能包含不同类型卡片的列表
            for card_group_wrapper in card_list:
                # 我们要找的帖子通常在card_group这个列表里
                if 'card_group' in card_group_wrapper:
                    for card in card_group_wrapper.get('card_group', []):
                        # 帖子卡片的类型通常是9
                        if card.get('card_type') == 9:
                            mblog = card.get('mblog', {})
                            if not mblog:
                                continue

                            # 安全地提取数据
                            user_info = mblog.get('user', {})
                            user_name = user_info.get('screen_name', '未知用户')
                            
                            # 提取并清洗正文，因为正文里有HTML标签
                            text_with_html = mblog.get('text', '')
                            soup = BeautifulSoup(text_with_html, 'lxml')
                            clean_text = soup.get_text(strip=True)

                            if clean_text:
                                scraped_data.append({
                                    'post_id': mblog.get('id', ''),
                                    'created_at': mblog.get('created_at', ''),
                                    'user_name': user_name,
                                    'text': clean_text
                                })
                                
        except Exception as e:
            print(f"\n解析第 {page} 页时发生未知错误: {e}")
            print("返回的原始文本:", response.text[:200])
            continue

        # f. 礼貌地请求，随机暂停
        sleep_time = random.uniform(2, 5)
        time.sleep(sleep_time)

    if scraped_data:
        return pd.DataFrame(scraped_data)
    else:
        return None

# --- 3. 执行脚本 ---
if __name__ == '__main__':
    SEARCH_KEYWORD = "上海地铁"
    TOTAL_PAGES = 10 
    OUTPUT_FILENAME = f"{SEARCH_KEYWORD}_data.csv"

    print(f"--- 开始采集任务 ---")
    
    df_result = scrape_weibo_data(SEARCH_KEYWORD, TOTAL_PAGES)
    
    if df_result is not None and not df_result.empty:
        df_result.to_csv(OUTPUT_FILENAME, index=False, encoding='utf-8-sig')
        print(f"\n--- 采集任务完成 ---")
        print(f"成功采集到 {len(df_result)} 条数据，已保存至文件: {OUTPUT_FILENAME}")
    else:
        print(f"\n--- 采集任务结束 ---")
        print("未能采集到任何有效数据。请最后一次检查：")
        print("1. HEADERS中的Cookie是否为最新。")
        print("2. API_URL和params中的containerid是否正确。")

