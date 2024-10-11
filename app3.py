import streamlit as st
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
import platform
import pymysql
import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# .env 파일에서 값을 가져옴
#host = os.getenv('DB_HOST')
#port = int(os.getenv('DB_PORT'))
#username = os.getenv('DB_USER')
#password = os.getenv('DB_PASSWORD')
#database = os.getenv('DB_NAME')

# secrets.toml의 값 불러오기
host = st.secrets["DB_HOST"]
port = int(st.secrets["DB_PORT"])
username = st.secrets["DB_USER"]
password = st.secrets["DB_PASS"]
database = st.secrets["DB_NAME"]

# 날짜에 따라 '순'을 구하는 함수
def get_순(purchase_date):
    day = purchase_date.day
    if day <= 10:
        return 'Early'
    elif 10 < day <= 20:
        return 'Mid'
    else:
        return 'Late'
    
def connect_to_db(host, port, username, password, database):
    try:
        connection = pymysql.connect(
            host=host,
            port=port,
            user=username,
            password=password,
            database=database,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        print("Successfully connected to the database!")
        return connection

    except pymysql.MySQLError as e:
        print(f"Error connecting to MySQL: {e}")
        return None

def fetch_data(connection, item):
    try:
        with connection.cursor() as cursor:
            item = item
            # cabbage, dried_pepper, garlic, ginger, green_onions, red_pepper
            sql_query = f"SELECT * FROM {item};"
            cursor.execute(sql_query)
            result = cursor.fetchall()
            df = pd.DataFrame(result)
            return df
    except:
        print("Error fetching data from the database!")
        return None
    
def get_date_range_value(purchase_date):
    year = purchase_date.year
    month = purchase_date.month
    day = purchase_date.day
    
    # 상순, 중순, 하순 구분
    if 1 <= day <= 10:
        순 = 0  # 상순
    elif 11 <= day <= 20:
        순 = 1  # 중순
    else:
        순 = 2  # 하순
    
    # 월을 10월부터 12월까지만 고려하므로 10월 상순은 0부터 시작
    if year == 2024 and month in [10, 11, 12]:
        base_value = (month - 10) * 3  # 각 월에 상순, 중순, 하순이 있으므로 월의 값에 3을 곱해줌
        final_value = base_value + 순
        return final_value
    else:
        raise ValueError("지원하는 날짜 범위는 2024년 10월부터 12월까지입니다.")
    
def map_items_to_korean(item):
    # 영문 테이블명과 한글 품목명 매핑
    item_mapping = {
        'cabbage_prediction': '배추',
        'dried_pepper_prediction': '건고추',
        'garlic_prediction': '깐마늘',
        'ginger_prediction': '생강',
        'green_onions_prediction': '쪽파',
        'red_pepper_prediction': '고춧가루'
    }
    return item_mapping.get(item, "알 수 없는 품목")

if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':  # macOS
    plt.rc('font', family='AppleGothic')
else:  # Linux
    plt.rc('font', family='NanumGothic')

# 마이너스 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="살래? 말래?", layout="wide")

# 커스텀 CSS 추가
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1E90FF;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #4682B4;
        margin-bottom: 1rem;
    }
    .info-text {
        font-size: 1rem;
        color: #708090;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>살래? 말래?</h1>", unsafe_allow_html=True)
st.markdown("<p class='info-text'>김장철 농산물 가격 비교 서비스</p>", unsafe_allow_html=True)
st.markdown("---")

# 사이드바 구성
with st.sidebar:
    st.markdown("<h2 class='sub-header'>입력 정보</h2>", unsafe_allow_html=True)
    
    ingredients = st.multiselect(
        "재료 선택",
        ["배추", "건고추", "깐마늘", "생강", "고춧가루", "쪽파"],
        default=["배추"]
    )
    
    purchase_date = st.date_input(
        "김장 예정일",
        min_value=date.today()
    )
    
    # 각 재료에 대한 슬라이더 생성
    ingredient_weights = {}
    for ingredient in ingredients:
        if ingredient == "배추":
            ingredient_weights[ingredient] = st.slider(
                f"{ingredient} (포기)",
                min_value=0,
                max_value=50,
                value=1,
                step=1
            )
        else:
            ingredient_weights[ingredient] = st.slider(
                f"{ingredient} 무게 (kg)",
                min_value=0.0,
                max_value=50.0,
                value=1.0,
                step=0.1
            )
    
    compare_button = st.button("비교하기")

################# 입력받은 purchase_date에 맞는 가격 정보 가져오기 #################
connection = connect_to_db(host, port, username, password, database)
# 데이터 가져오기
list = ['cabbage_prediction', 'dried_pepper_prediction', 'garlic_prediction', 'ginger_prediction', 'green_onions_prediction', 'red_pepper_prediction']
ingredient_prices = {}
for item in list:
    print('-'*50)
    print(f"Processing {item} data...")
    print('-'*50)

    # 
    temp = get_date_range_value(purchase_date)

    if connection:
        korean_name = map_items_to_korean(item) # 품목명 한글로 변환
        fetched_data = fetch_data(connection, item=item)  # 데이터 가져오기
        if not fetched_data.empty:
            price = fetched_data.iloc[temp]['price']  # temp에 맞는 가격을 가져옴
            ingredient_prices[korean_name] = price
        else:
            print(f"{item}에서 데이터를 찾을 수 없습니다.")

# 메인 페이지 구성
if compare_button:
    st.markdown("<h2 class='sub-header'>비교 결과</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3 class='sub-header'>1. 직접 담그기</h3>", unsafe_allow_html=True)
        
        estimated_cost = sum(ingredient_prices[ing] * weight for ing, weight in ingredient_weights.items())
        st.markdown(f"<div class='highlight'><h4>예상 비용: {estimated_cost:,.0f}원</h4></div>", unsafe_allow_html=True)
        
        # 선택된 재료별 가격 표시
        prices = [ingredient_prices[ing] for ing in ingredients]
        
        fig, ax = plt.subplots()
        ax.bar(ingredients, prices, color='skyblue')
        ax.set_ylabel('가격 (원)')
        ax.set_title('선택된 재료별 가격')
        
        for i, v in enumerate(prices):
            ax.text(i, v + 100, f"{int(v):,}원", ha='center', va='bottom')
        
        st.pyplot(fig)
        
        st.write(f"📅 김장 예정일: {purchase_date}")
    
    with col2:
        st.markdown("<h3 class='sub-header'>2. 포장 김치 구매</h3>", unsafe_allow_html=True)
        
        # 총 김치 무게 계산 (배추는 포기당 3kg으로 계산)
        total_weight = 0
        for ingredient, weight in ingredient_weights.items():
            if ingredient == "배추":
                total_weight += weight * 3  # 배추는 포기당 3kg으로 계산
            else:
                pass
                #total_weight += weight  # 다른 재료는 kg으로 환산하지 않음
        
        df = pd.DataFrame([
            {"브랜드": "종가집", "제품명": "포기김치", "중량": f"{total_weight}kg", 
             "가격": f"{14970 * total_weight / 1:,.0f}원"},
            {"브랜드": "비비고", "제품명": "포기김치", "중량": f"{total_weight}kg", 
             "가격": f"{15900 * total_weight / 1:,.0f}원"},
            {"브랜드": "이마트 노브랜드", "제품명": "별미 포기김치", "중량": f"{total_weight}kg", 
             "가격": f"{19720 * total_weight / 3.5:,.0f}원"},
            {"브랜드": "늘만나김치", "제품명": "맛김치", "중량": f"{total_weight}kg", 
             "가격": f"{7000 * total_weight / 0.45:,.0f}원"}
        ])
        
        st.table(df)

        packaged_kimchi = [
            {"name": "종가집 포기김치", "url": "https://www.ssg.com/item/itemView.ssg?itemId=1000012927413"},
            {"name": "비비고 포기김치", "url": "https://www.cjthemarket.com/pc/prod/prodDetail?prdCd=40145356"},
            {"name": "이마트 노브랜드 별미 포기김치", "url": "https://emart.ssg.com/search.ssg?target=all&query=%ED%8F%AC%EA%B8%B0%EA%B9%80%EC%B9%98"},
            {"name": "늘만나김치 맛김치", "url": "https://www.mannakimchi.com"},
        ]
        
        st.markdown("<h4>추천 제품 링크</h4>", unsafe_allow_html=True)
        
        for kimchi in packaged_kimchi:
            st.markdown(f"- [{kimchi['name']}]({kimchi['url']})")

else:
    st.info("👈 왼쪽 사이드바에서 정보를 입력하고 '비교하기' 버튼을 눌러주세요.")