import requests
import pandas as pd
import time
import random
from tqdm import tqdm
from bs4 import BeautifulSoup
import json
import os

# --- 1. 配置区域 ---

API_URL = "https://m.weibo.cn/api/container/getIndex?containerid=100103type%3D1%26q%3D%E4%B8%8A%E6%B5%B7%E5%9C%B0%E9%93%81+-%E6%96%B0%E9%97%BB&page_type=searchall"

# TODO: (必须替换) 每次运行前，从浏览器F12开发者工具中获取最新的Cookie
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36',
    'Cookie': 'WEIBOCN_FROM=1110006030; BAIDU_SSP_lcr=https://www.google.com/; _T_WM=79542187288; XSRF-TOKEN=40295d; mweibo_short_token=20193f73f0; MLOGIN=0; M_WEIBOCN_PARAMS=fid%3D100103type%253D1%2526q%253D%25E4%25B8%258A%25E6%25B5%25B7%25E5%259C%25B0%25E9%2593%2581%2B-%25E6%2596%25B0%25E9%2597%25BB%26uicode%3D10000011', # 请务必替换成你自己的完整、最新Cookie！
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7',
    'Connection': 'keep-alive',
    'MWeibo-Pwa': '1',
    'X-Requested-With': 'XMLHttpRequest',
    'X-XSRF-TOKEN': '40295d',
}

# --- 2. 智能过滤器函数 ---
def is_high_quality_ugc(text, username):
    """判断一条文本是否为高质量的用户生成内容(UGC)。"""
    if not isinstance(text, str) or not isinstance(username, str):
        return False
    # 规则1：屏蔽新闻标题格式
    if text.strip().startswith('【'):
        return False
    # 规则2：屏蔽新闻/营销关键词
    noise_keywords = ['记者获悉', '小编', '本报讯', '官方', '公告', '活动', '直播', '试点', '扩围', '据悉', '发布']
    if any(keyword in text for keyword in noise_keywords):
        return False
    # 规则3：屏蔽官方或媒体账号
    official_accounts = ['上海地铁shmetro', '央视新闻', '人民日报', '头条新闻', '新华社', '澎湃新闻', '观察者网']
    if username in official_accounts:
        return False
    # 规则4：屏蔽过短的文本
    if len(text.strip()) < 10:
        return False
    return True


# --- 3. 主采集函数 ---
def scrape_weibo_data(keyword, pages_to_scrape=10, start_page=1):
    """根据关键词、要爬取的页数和起始页码，爬取并过滤微博数据。"""
    scraped_data = []
    total_filtered = 0
    
    end_page = start_page + pages_to_scrape
    for page in tqdm(range(start_page, end_page), desc=f"采集'{keyword}' (Pages {start_page}-{end_page-1})"):
        params = {'containerid': f'100103type=1&q={keyword}', 'page_type': 'searchall', 'page': page}
        
        try:
            response = requests.get(API_URL, headers=HEADERS, params=params, timeout=20)
            response.raise_for_status() 
            data = response.json()
            card_list = data.get('data', {}).get('cards', [])
            
            if not card_list:
                print(f"\n在第 {page} 页未找到'cards'列表，可能已到达末页或被限制，提前结束本次任务。")
                break

            for card_group_wrapper in card_list:
                if 'card_group' in card_group_wrapper:
                    for card in card_group_wrapper.get('card_group', []):
                        if card.get('card_type') == 9:
                            mblog = card.get('mblog', {})
                            if not mblog: continue

                            user_info = mblog.get('user', {})
                            user_name = user_info.get('screen_name', '')
                            text_with_html = mblog.get('text', '')
                            
                            soup = BeautifulSoup(text_with_html, 'lxml')
                            clean_text = soup.get_text(strip=True)

                            if is_high_quality_ugc(clean_text, user_name):
                                scraped_data.append({
                                    'post_id': mblog.get('id', ''),
                                    'created_at': mblog.get('created_at', ''),
                                    'user_name': user_name,
                                    'text': clean_text
                                })
                            else:
                                total_filtered += 1
        except Exception as e:
            print(f"\n处理第 {page} 页时发生错误: {e}")
            
        time.sleep(random.uniform(3, 6))

    print(f"\n在本次批次中，共过滤掉了 {total_filtered} 条低质量或无关数据。")
    
    if scraped_data:
        return pd.DataFrame(scraped_data)
    else:
        return None

# --- 4. 智能执行与保存逻辑 ---
if __name__ == '__main__':
    # --- 任务配置 ---
    SEARCH_KEYWORD = "上海 网约车"
    OUTPUT_FILENAME = "data/weibo_data_master.csv"

    # --- 本次运行的参数 ---
    # 每次运行时，只需要修改这里的起始页和要爬取的页数
    # 例如，第一次跑：START_PAGE = 1, PAGES_TO_SCRAPE = 100
    # 第二次跑：START_PAGE = 101, PAGES_TO_SCRAPE = 100
    START_PAGE = 1
    PAGES_TO_SCRAPE = 100

    print(f"--- 开始批次采集任务 ---")
    print(f"关键词: '{SEARCH_KEYWORD}', 计划采集页码: {START_PAGE} 到 {START_PAGE + PAGES_TO_SCRAPE - 1}")
    
    # 执行采集，获取本次运行的新数据
    df_new_data = scrape_weibo_data(SEARCH_KEYWORD, PAGES_TO_SCRAPE, START_PAGE)
    
    if df_new_data is None or df_new_data.empty:
        print("本次运行未能采集到任何新的有效数据。任务结束。")
    else:
        # 检查主文件是否已存在
        if os.path.exists(OUTPUT_FILENAME):
            print(f"检测到已存在的主文件'{OUTPUT_FILENAME}'，将进行数据追加。")
            df_old_data = pd.read_csv(OUTPUT_FILENAME)
            # 使用pd.concat合并新旧数据
            df_combined = pd.concat([df_old_data, df_new_data], ignore_index=True)
            print(f"合并前总数据量: {len(df_combined)}")
            # 根据帖子ID去重，保留最新的一次采集
            df_combined.drop_duplicates(subset=['post_id'], keep='last', inplace=True)
            print(f"合并去重后总数据量: {len(df_combined)}")
        else:
            print("未找到主文件，将创建新文件。")
            df_combined = df_new_data

        df_combined.to_csv(OUTPUT_FILENAME, index=False, encoding='utf-8-sig')
        print(f"数据已成功保存或更新至 {OUTPUT_FILENAME}")