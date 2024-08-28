import streamlit as st
import yfinance as yf
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta  # datetime 모듈 임포트
import matplotlib.pyplot as plt
from PIL import Image
import os
import itertools
import requests
from bs4 import BeautifulSoup
from io import BytesIO
from matplotlib.patches import Ellipse
import openai
from pandas_datareader import data as pdr

import warnings




api_key = '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'
openai.api_key = api_key

if 'portfolio_data' not in st.session_state:
    st.session_state.portfolio_data = pd.DataFrame(columns=['month', 'dividend', 'ticker'])

def download_logo_to_file(url, filename):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        img.save(filename)
    except Exception as e:
        st.error(f"이미지를 다운로드하는 중 오류 발생: {e}")

# 주식 배당금 데이터를 가져와 포트폴리오에 추가하는 함수
def stack_data(ticker, num_of_shares):
    stock = yf.Ticker(ticker)
    dividends_df = stock.dividends.reset_index()
    dividends_df['Date'] = pd.to_datetime(dividends_df['Date'])
    dividends_df = dividends_df[dividends_df['Date'] > '2020-08-12']
    dividends_df['month'] = dividends_df['Date'].dt.month
    dividends_df['ticker'] = ticker
    dividends_df = dividends_df[['month', 'Dividends', 'ticker']]  # 필요한 열만 선택
    dividends_df.rename(columns={'Dividends': 'dividend'}, inplace=True)
    dividends_df['dividend'] *= num_of_shares
    return dividends_df

# GPT에서 티커 정보를 가져오는 함수
def get_ticker(query):
    model = "gpt-4o-mini"
    messages = [
        {"role": "system", "content": "입력받은 주제와 관련된 기업의 미국 주식 티커를 알려줘, 티커인 알파벳 1~5자리로만 말해야 해. 예를들면 KO, AAPL처럼 따옴표 출력도 하지 말고 오로지 티커 1~5자리만 뱉아. 만약 가장 맛없는 콜라 기업을 질문받으면 PEP, 만약 이해할 수 없는 설명을 입력받으면 KO를 출력하도록 해."},
        {"role": "user", "content": query}
    ]
    response = openai.ChatCompletion.create(model=model, messages=messages)
    return response.choices[0].message['content'].strip()

# 캔들차트를 그리는 함수
def create_candlestick_chart(df):
    fig = go.Figure(data=[go.Candlestick(
                x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
            )])
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        width=900,
        height=600
    )
    return fig

# 포트폴리오 데이터 초기화
def initialize_portfolio():
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = pd.DataFrame(columns=['ticker', 'name', 'num_shares', 'dividends', 'total_investment'])

# 초기 자본 계산 함수
def calculate_remaining_capital(initial_capital):
    if 'remaining_capital' not in st.session_state:
        st.session_state.remaining_capital = initial_capital
    else:
        st.session_state.remaining_capital = initial_capital - st.session_state.portfolio['total_investment'].sum()

# 주식 데이터 가져오기 및 포트폴리오 업데이트 함수
def add_stock_to_portfolio(ticker, stock, latest_price, investment_amount, latest_dividend):
    # 기존에 같은 티커가 있는지 확인
    existing_row = st.session_state.portfolio[st.session_state.portfolio['ticker'] == ticker]

    if not existing_row.empty:
        # 기존 티커가 있는 경우, 수량, 배당금, 총 투자금 업데이트
        index = existing_row.index[0]
        st.session_state.portfolio.at[index, 'num_shares'] += investment_amount
        st.session_state.portfolio.at[index, 'dividends'] += (latest_dividend * investment_amount if latest_dividend is not None else 0)
        st.session_state.portfolio.at[index, 'total_investment'] += (investment_amount * latest_price)
        st.success(f'{stock.info["shortName"]} 주식 {investment_amount} 주가 포트폴리오에 추가되었습니다. (기존 티커 업데이트)')
    else:
        # 새로운 티커인 경우, 새로운 행 추가
        new_entry = {
            'ticker': ticker,
            'name': stock.info["shortName"],
            'num_shares': investment_amount,
            'dividends': latest_dividend * investment_amount if latest_dividend is not None else 0,
            'total_investment': investment_amount * latest_price
        }
        new_entry_df = pd.DataFrame([new_entry])
        st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_entry_df], ignore_index=True)
        st.success(f'{new_entry["name"]} 주식 {investment_amount} 주가 포트폴리오에 추가되었습니다.')

    # 남은 자본 업데이트
    st.session_state.remaining_capital -= investment_amount * latest_price


tab1, tab2, tab3 = st.tabs(["나만의 배당주 포트폴리오 구성하기","포트폴리오 만들기", "포트폴리오 재투자 금액 차이 분석"])

with tab1:
    st.title("나만의 배당주 포트폴리오 구성하기")
    # 배당킹주와 배당귀족주 리스트 및 배당 월 정보
    dividend_king_stocks = {
        'MMM': ('3M', 'https://seeklogo.com/images/1/3M-logo-DCF26CFF14-seeklogo.com.png', [3, 6, 9, 12]),
        'KO': ('Coca-Cola', 'https://seeklogo.com/images/C/coca-cola-circle-logo-A9EBD3B00A-seeklogo.com.png', [1, 4, 7, 10]),
        'JNJ': ('Johnson & Johnson', 'https://seeklogo.com/images/J/johnson-johnson-logo-5912A7508E-seeklogo.com.png', [2, 5, 8, 11]),
        'PG': ('Procter & Gamble', 'https://seeklogo.com/images/P/p-g-logo-14BC19B5E7-seeklogo.com.png', [3, 6, 9, 12]),
        'CL': ('Colgate-Palmolive', 'https://seeklogo.com/images/C/colgate-palmolive-logo-63D53420E1-seeklogo.com.png', [1, 4, 7, 10]),
        'PEP': ('PepsiCo', 'https://seeklogo.com/images/P/pepsi-vertical-logo-72846897FF-seeklogo.com.png', [3, 6, 9, 12]),
        'GPC': ('Genuine Parts Company', 'https://seeklogo.com/images/G/genuine-parts-company-logo-23D67B3040-seeklogo.com.png', [3, 6, 9, 12]),
        'ABT': ('Abbott', 'https://g.foolcdn.com/art/companylogos/mark/ABT.png', [2, 5, 8, 11]),
        'PH': ('Parker Hannifin', 'https://seeklogo.com/images/P/Parker_Hannifin-logo-30D7790AEF-seeklogo.com.png', [2, 5, 8, 11]),
        'WBA': ('Walgreens Boots Alliance', 'https://seeklogo.com/images/W/Walgreens-logo-38D42E4EC1-seeklogo.com.png', [1, 4, 7, 10]),
        'LOW': ('Lowe’s', 'https://seeklogo.com/images/L/lowes-logo-BD8C045F2F-seeklogo.com.png', [2, 5, 8, 11]),
        'CLX': ('Clorox', 'https://seeklogo.com/images/C/Clorox-logo-88E4ED3C26-seeklogo.com.png', [3, 6, 9, 12]),
        'HRL': ('Hormel Foods', 'https://seeklogo.com/images/H/hormel-logo-2C9BC6463A-seeklogo.com.png', [1, 4, 7, 10]),
        'CVX': ('Chevron', 'https://seeklogo.com/images/C/Chevron_Corporation-logo-FFAC2E8206-seeklogo.com.png', [2, 5, 8, 11]),
        'EMR': ('Emerson Electric', 'https://seeklogo.com/images/E/Emerson_Electric-logo-CF7EACA482-seeklogo.com.png', [3, 6, 9, 12]),
        'SYY': ('Sysco', 'https://seeklogo.com/images/S/sysco-logo-7B5B009D80-seeklogo.com.png', [1, 4, 7, 10]),
        'SWK': ('Stanley Black & Decker', 'https://seeklogo.com/images/S/stanley-black-decker-logo-81E59F852A-seeklogo.com.png', [2, 5, 8, 11]),
        'AFL': ('Aflac', 'https://seeklogo.com/images/A/AFLAC-logo-EDE6C89650-seeklogo.com.png', [3, 6, 9, 12]),
        'SHW': ('Sherwin-Williams', 'https://seeklogo.com/images/S/Sherwin_Williams-logo-3FE71297BA-seeklogo.com.png', [1, 4, 7, 10]),
        'LLY': ('Eli Lilly', 'https://seeklogo.com/images/L/Lilly-logo-6EF04E4361-seeklogo.com.png', [2, 5, 8, 11]),
    }
    dividend_aristocrat_stocks = {
        'ADM': ('Archer Daniels Midland', 'https://seeklogo.com/images/A/archer-daniels-midland-company-logo-B4F247583E-seeklogo.com.png', [1, 4, 7, 10]),
        'ABBV': ('AbbVie', 'https://seeklogo.com/images/A/abbvie-logo-BEB7C12577-seeklogo.com.png', [2, 5, 8, 11]),
        'DOV': ('Dover', 'https://seeklogo.com/images/D/Dover-logo-2F344F6F42-seeklogo.com.png', [3, 6, 9, 12]),
        'ITW': ('Illinois Tool Works', 'https://seeklogo.com/images/I/illinois-tool-works-logo-FCB6FE9266-seeklogo.com.png', [1, 4, 7, 10]),
        'KMB': ('Kimberly-Clark', 'https://seeklogo.com/images/K/Kimberly-Clark_Sopalin-logo-8B40BD9217-seeklogo.com.png', [3, 6, 9, 12]),
        'XOM': ('ExxonMobil', 'https://seeklogo.com/images/E/Exxon-logo-6F21C176C8-seeklogo.com.png', [1, 4, 7, 10]),
        'MDT': ('Medtronic', 'https://seeklogo.com/images/M/medtronic-healthcare-logo-97942C1A14-seeklogo.com.png', [2, 5, 8, 11]),
        'WMT': ('Walmart', 'https://seeklogo.com/images/W/walmart-spark-logo-57DC35C86C-seeklogo.com.png', [1, 4, 7, 10]),
        'TROW': ('T. Rowe Price', 'https://seeklogo.com/images/T/trow-logo-5361227321-seeklogo.com.png', [2, 5, 8, 11]),
        'APD': ('Air Products and Chemicals', 'https://seeklogo.com/images/A/Air_Products_and_Chemicals-logo-ACDA8A1C8B-seeklogo.com.png', [1, 4, 7, 10]),
        'BF-B': ('Brown-Forman', 'https://seeklogo.com/images/B/brown-forman-logo-B919105D7B-seeklogo.com.png', [2, 5, 8, 11]),
        'CINF': ('Cincinnati Financial', 'https://seeklogo.com/images/C/cincinnati-financial-logo-A2A7957DB1-seeklogo.com.png', [3, 6, 9, 12]),
        'CTAS': ('Cintas', 'https://seeklogo.com/images/C/cintas-logo-0F4637C8B8-seeklogo.com.png', [1, 4, 7, 10]),
        'ED': ('Consolidated Edison', 'https://seeklogo.com/images/C/consolidated-edison-logo-F23BE97D80-seeklogo.com.png', [2, 5, 8, 11]),
        'GWW': ('Grainger', 'https://seeklogo.com/images/G/grainger-logo-C959D21C07-seeklogo.com.png', [1, 4, 7, 10]),
        'MKC': ('McCormick', 'https://seeklogo.com/images/M/McCormick-logo-144428A8DB-seeklogo.com.png', [3, 6, 9, 12]),
        'NUE': ('Nucor', 'https://seeklogo.com/images/N/Nucor-logo-E63140A596-seeklogo.com.png', [2, 5, 8, 11]),
        'ROP': ('Roper Technologies', 'https://seeklogo.com/images/R/Roper-logo-73AE49CBF0-seeklogo.com.png', [3, 6, 9, 12]),
        'SPGI': ('S&P Global', 'https://seeklogo.com/images/S/s-p-global-logo-62660CED63-seeklogo.com.png', [1, 4, 7, 10]),
        'TGT': ('Target', 'https://seeklogo.com/images/T/Target-logo-9FE48EBE3B-seeklogo.com.png', [2, 5, 8, 11]),
        'MCD': ('McDonald’s', 'https://seeklogo.com/images/M/mcdonald-s-golden-arches-logo-93483062BF-seeklogo.com.png', [3, 6, 9, 12]),
    }
    if 'selected_companies' not in st.session_state:
        st.session_state['selected_companies'] = []
    # Create placeholder for chart area
    chart_placeholder = st.empty()
    # Function to draw the chart
    def draw_chart():
        fig, ax = plt.subplots(figsize=(12, 7))  # Adjust chart size
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(months, fontsize=12, rotation=45)  # Rotate x-axis labels and adjust font size
        ax.set_xlim(0.5, 12.5)
        ax.set_ylim(-0.5, max(5, len(st.session_state.selected_companies)))  # Dynamic y-axis size
        ax.grid(True, linestyle='--', alpha=0.6)  # Adjust grid style
        ax.set_title("Dividend Portfolio", fontsize=18, fontweight='bold', pad=20)  # Add chart title
        ax.legend(loc='upper right', fontsize=12)
        # Add logos for selected stocks
        if st.session_state.selected_companies:
            company_names = [
                (dividend_king_stocks[company][0] if category == 'king' else dividend_aristocrat_stocks[company][0])
                for company, category in st.session_state.selected_companies
            ]
            ax.set_yticks(range(len(st.session_state.selected_companies)))
            ax.set_yticklabels(company_names, fontsize=12, fontweight='bold')  # Adjust y-axis label style
            for idx, (company, category) in enumerate(st.session_state.selected_companies):
                dividend_months = (dividend_king_stocks[company][2] if category == 'king' else dividend_aristocrat_stocks[company][2])
                logo_url = (dividend_king_stocks[company][1] if category == 'king' else dividend_aristocrat_stocks[company][1])
                logo_filename = f"{company}_logo.png"
                # Download logo if it doesn't exist
                if not os.path.exists(logo_filename):
                    download_logo_to_file(logo_url, logo_filename)
                # Load logo image from local file
                if os.path.exists(logo_filename):
                    logo_image = Image.open(logo_filename).resize((40, 40))  # Resize
                    # Display logo on the chart
                    for month in dividend_months:
                        ax.imshow(logo_image, extent=(month-0.35, month+0.35, idx-0.25, idx+0.25), aspect='auto')
        # Update chart in the placeholder
        chart_placeholder.pyplot(fig)
        plt.close(fig)  # Close the chart to prevent memory leaks
    # Create a mapping of tickers to company names
    ticker_to_name_king = {ticker: name for ticker, (name, _, _) in dividend_king_stocks.items()}
    ticker_to_name_aristocrat = {ticker: name for ticker, (name, _, _) in dividend_aristocrat_stocks.items()}
    king_selection = st.selectbox("배당킹주에서 선택", ['선택하세요'] + list(ticker_to_name_king.values()), key='select_king')
    if st.button("추가", key='add_king') and king_selection != '선택하세요':
        ticker = [ticker for ticker, name in ticker_to_name_king.items() if name == king_selection][0]
        st.session_state.selected_companies.append((ticker, 'king'))
        draw_chart()  # Update the chart
    aristocrat_selection = st.selectbox("배당귀족주에서 선택", ['선택하세요'] + list(ticker_to_name_aristocrat.values()), key='select_aristocrat')
    if st.button("추가", key='add_aristocrat') and aristocrat_selection != '선택하세요':
        ticker = [ticker for ticker, name in ticker_to_name_aristocrat.items() if name == aristocrat_selection][0]
        st.session_state.selected_companies.append((ticker, 'aristocrat'))
        draw_chart()  # Update the chart
    # Remove stock from portfolio
    remove_company = st.selectbox("제거할 주식 선택", ['선택하세요'] + [ticker_to_name_king.get(company, '') for company, _ in st.session_state.selected_companies] + [ticker_to_name_aristocrat.get(company, '') for company, _ in st.session_state.selected_companies], key='remove_select')
    if st.button("제거", key='remove_company') and remove_company != '선택하세요':
        ticker_to_remove = [ticker for ticker, name in ticker_to_name_king.items() if name == remove_company] + [ticker for ticker, name in ticker_to_name_aristocrat.items() if name == remove_company]
        st.session_state.selected_companies = [(company, category) for company, category in st.session_state.selected_companies if company not in ticker_to_remove]
        st.success(f"{remove_company}이(가) 포트폴리오에서 제거되었습니다.")
        draw_chart()  # Update the chart
    # Display an empty chart initially
    draw_chart()


with tab2:
    st.title('나만의 월 배당 포트폴리오 구축')

    initialize_portfolio()

    initial_capital = st.number_input('초기 자본을 입력하세요 (단위: 달러)', min_value=0, value=100000, step=1000)
    calculate_remaining_capital(initial_capital)

    st.write(f"남은 자본: ${st.session_state.remaining_capital:.2f}")

    query = st.text_input('주식을 검색하세요', placeholder='Ex) 애플, 코카콜라')
    if query:
        if 'ticker' not in st.session_state or st.session_state.query != query:
            try:
                ticker = get_ticker(query)
                st.session_state.ticker = ticker
                st.session_state.query = query
            except Exception as e:
                st.error(f"GPT에서 티커를 가져오는 중 오류 발생: {e}")
                ticker = None
        else:
            ticker = st.session_state.ticker

        if ticker:
            try:
                stock = yf.Ticker(ticker)
                stock_data = stock.history(period="2y").reset_index()
                dividends = stock.dividends if not stock.dividends.empty else None

                if not stock_data.empty:
                    latest_price = stock_data['Close'].iloc[-1]
                    st.write(f'현재 {stock.info["shortName"]}의 주가는 ${latest_price:.2f}입니다.')
                    st.plotly_chart(create_candlestick_chart(stock_data))

                    if dividends is not None:
                        latest_dividend = dividends.iloc[-1]
                        st.write(f'현재 {stock.info["shortName"]}의 최신 배당금: ${latest_dividend:.2f}')
                    else:
                        st.write(f'{stock.info["shortName"]}의 배당금 정보가 없습니다.')

                    investment_amount = st.slider('매수할 주식 수', value=0, min_value=0, max_value=int(st.session_state.remaining_capital / latest_price))
                    st.write(f"매수 후 남은 자본: ${st.session_state.remaining_capital - investment_amount * latest_price:.2f}")

                    if st.button("매수"):
                        add_stock_to_portfolio(ticker, stock, latest_price, investment_amount, latest_dividend)
                        # 티커를 포트폴리오 데이터에 추가 (세션 상태에 저장)
                        new_portfolio_data = stack_data(ticker, investment_amount)
                        st.session_state.portfolio_data = pd.concat([st.session_state.portfolio_data, new_portfolio_data], ignore_index=True)

                else:
                    st.error(f'{stock.info["shortName"]} 주식에 대한 데이터를 가져올 수 없습니다.')
            except Exception as e:
                st.error(f"데이터를 가져오는 중 오류 발생: {e}")

    # 포트폴리오에 배당금 정보가 있을 때 월별 배당금 그래프 표시

    if not st.session_state.portfolio_data.empty:
        months = [str(i) for i in range(1, 13)]

        fig = px.bar(
            st.session_state.portfolio_data,
            x='month',
            y='dividend',
            color='ticker',
            hover_data=['ticker', 'dividend'],
            labels={'month': '월', 'dividend': '배당금 ($)'},
            title='포트폴리오의 월별 배당금 흐름',
            barmode='stack',
            category_orders={"month": months}
        )

        # X축의 틱 값을 명시적으로 설정하여 1~12월까지 모두 표시되도록 함
        fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=list(range(1, 13)),
                ticktext=months
            )
        )

        st.plotly_chart(fig)
    else:
        st.write("포트폴리오에 배당금 정보가 없습니다.")

    # 인덱스를 제거하여 포트폴리오 DataFrame 출력
    st.dataframe(st.session_state.portfolio.reset_index(drop=True))
with tab3:
    # CSV 파일 경로
    CSV_FILE_PATH = 'ticker_data.csv'

    # 데이터 로드
    @st.cache_data
    def load_data():
        return pd.read_csv(CSV_FILE_PATH)

    df = load_data()

    # Streamlit 앱 제목
    st.title('SCHD ETF Companies by Sector')

    # 데이터 미리보기
    st.subheader('Data Preview')
    st.write(df.head())

    # 트리맵 시각화
    fig = px.treemap(df, 
                    path=['Sector', 'Name'], 
                    values='MarketCap',
                    title='SCHD ETF Companies Treemap by Sector')

    st.plotly_chart(fig)

    dividend_stocks = ['AAPL', 'MSFT', 'KO']  # 예시: 배당주 티커
    non_dividend_stocks = ['GOOGL', 'AMZN', 'TSLA']  # 예시: 비배당주 티커

    # 데이터 수집 함수
    def get_stock_data(tickers, start_date, end_date):
        data = {}
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date, actions=True)
            df['Ticker'] = ticker
            data[ticker] = df
        return pd.concat(data.values())

    # 데이터 가져오기
    start_date = '2024-01-01'
    end_date = '2024-07-01'
    dividend_data = get_stock_data(dividend_stocks, start_date, end_date)
    non_dividend_data = get_stock_data(non_dividend_stocks, start_date, end_date)

    # 배당금 포함된 데이터 처리
    def calculate_total_return(df):
        df['Dividend'] = df['Dividends']
        df['Total Return'] = (df['Close'] + df['Dividend']) / df['Close'].shift(1) - 1
        return df

    # 배당주와 비배당주 데이터를 하나의 데이터프레임으로 결합
    dividend_data = calculate_total_return(dividend_data)
    non_dividend_data = calculate_total_return(non_dividend_data)
    combined_data = pd.concat([dividend_data, non_dividend_data])

    # Streamlit 애플리케이션
    st.subheader('Dividend vs Non-Dividend Stock Total Return Comparison')

    # Plotly 그래프 생성
    fig = px.line(
        combined_data, 
        x=combined_data.index, 
        y='Total Return', 
        color='Ticker', 
        labels={'Total Return': 'Total Return', 'index': 'Date'}, 
        title='Total Return Comparison'
    )

    fig.update_traces(opacity=0.6) # 선의 투명도 설정
    fig.update_yaxes(range=[-0.15, 0.15])
    st.plotly_chart(fig)

    import plotly.graph_objs as go

    def calculate_group_average(df, tickers):
        df_group = df[df['Ticker'].isin(tickers)]
        avg_return = df_group.groupby(df_group.index)['Total Return'].mean()
        return avg_return

    # 평균 수익률 계산
    dividend_avg_return = calculate_group_average(dividend_data, dividend_stocks)
    non_dividend_avg_return = calculate_group_average(non_dividend_data, non_dividend_stocks)
    combined_data2 = pd.concat([dividend_avg_return, non_dividend_avg_return])

    # Streamlit 애플리케이션
    st.subheader('Average Total Return Comparison: Dividend vs Non-Dividend Stocks')
    fig = go.Figure()

    # 배당주 평균 수익률 선 추가
    fig.add_trace(go.Scatter(
        x=dividend_avg_return.index, 
        y=dividend_avg_return, 
        mode='lines', 
        name='Dividend Stocks',
        line=dict(color='blue', width=2)  # 선 색상 및 너비 지정
    ))

    # 비배당주 평균 수익률 선 추가
    fig.add_trace(go.Scatter(
        x=non_dividend_avg_return.index, 
        y=non_dividend_avg_return, 
        mode='lines', 
        name='Non-Dividend Stocks',
        line=dict(color='red', width=2)  # 선 색상 및 너비 지정
    ))

    fig.update_traces(opacity=0.6)  # 선의 투명도 설정
    fig.update_yaxes(range=[-0.05, 0.05])
    # 그래프 제목 및 레이블 설정
    fig.update_layout(
        title='Average Total Return Comparison: Dividend vs Non-Dividend Stocks',
        xaxis_title='Date',
        yaxis_title='Total Return'
    )
    st.plotly_chart(fig)

    df = pd.read_csv("etf_companies_info.csv")

    # Group data by ETF
    etf_summary = df.groupby('ETF').agg({
        'AUM': 'mean',
        'Dividend Yield': 'mean',
        #'Expense Ratio': 'mean',
        'YTD Return': 'mean',
        '5-Year Avg Return': 'mean'
    }).reset_index()

    # Start Streamlit application
    st.title('ETF Performance Comparison Visualization')
    # Visualize ETF performance data
    st.subheader('Average AUM, Dividend Yield, Expense Ratio, and Returns by ETF')

    # AUM Visualization
    st.subheader('Average AUM by ETF')
    fig = px.bar(etf_summary, x='ETF', y='AUM', title='Average AUM by ETF', labels={'AUM': 'Average AUM ($)', 'ETF': 'ETF'}, color_discrete_sequence=['skyblue'])
    # Show plotly chart in Streamlit
    st.plotly_chart(fig)

    # Dividend Yield Visualization
    st.subheader('Average Dividend Yield by ETF')
    fig = px.bar(etf_summary, x='ETF', y='Dividend Yield', title='Average Dividend Yield by ETF', labels={'Dividend Yield':'Dividend Yield' , 'ETF': 'ETF'}, color_discrete_sequence=['#e377c2'])
    st.plotly_chart(fig)

    # YTD Return Visualization
    st.subheader('Average YTD Return by ETF')
    fig = px.bar(etf_summary, x='ETF', y='YTD Return', title='Average YTD Return by ETF', labels={'YTD Return': 'YTD Return', 'ETF': 'ETF'}, color_discrete_sequence=['#2ca02c'])
    st.plotly_chart(fig)

    # 5-Year Average Return Visualization
    st.subheader('Average 5-Year Avg Return by ETF')
    fig = px.bar(etf_summary, x='ETF', y='5-Year Avg Return', title='Average 5-Year Avg Return by ETF', labels={'5-Year Avg Return': '5-Year Avg Return', 'ETF': 'ETF'}, color_discrete_sequence=['#9467bd'])
    st.plotly_chart(fig)

    # CSV 파일 읽기
    df = pd.read_csv("etf_companies_info.csv")

    # Plot settings
    colors = {'VYM': 'blue', 'VIG': 'green', 'SCHD': 'red'}

    # Streamlit 애플리케이션 레이아웃 정의
    st.title('ETF Performance Scatter Plot Analysis')

    # X축 선택 옵션
    x_axis = st.selectbox(
        "Select X-axis:",
        options=['AUM', 'Dividend Yield', 'YTD Return', '5-Year Avg Return'],
        index=0
    )

    # Y축 선택 옵션
    y_axis = st.selectbox(
        "Select Y-axis:",
        options=['AUM', 'Dividend Yield', 'YTD Return', '5-Year Avg Return'],
        index=1
    )

    # Scatter plot 생성
    fig = px.scatter(
        df, 
        x=x_axis, 
        y=y_axis, 
        color='ETF',
        color_discrete_map=colors,
        hover_name='Company Name',  # Hover 시 종목 이름 표시
        labels={
            'AUM': 'AUM ($)',
            'Dividend Yield': 'Dividend Yield (%)',
            'Expense Ratio': 'Expense Ratio (%)',
            'YTD Return': 'YTD Return (%)',
            '5-Year Avg Return': '5-Year Avg Return (%)'
        }
    )

    fig.update_layout(title=f'Scatter Plot of {x_axis} vs {y_axis} by ETF')
    st.plotly_chart(fig)


    # Streamlit 설정
    st.title("Coca-Cola Stock Price vs. Inflation Rate Over Time")

    # 데이터 수집
    ticker = 'KO'
    start_date = "2010-01-01"
    end_date = datetime.today().strftime('%Y-%m-%d')

    # 코카콜라 주가 데이터 가져오기
    ko_data = yf.download(ticker, start=start_date, end=end_date)
    ko_data['Price Change %'] = ko_data['Adj Close'].pct_change() * 100

    # 인플레이션율 데이터 (CPI 데이터)
    cpi_data = pd.read_csv('CPIAUCSL.csv', parse_dates=['DATE'], index_col='DATE')
    cpi_data = cpi_data.rename(columns={'VALUE': 'CPIAUCSL'})

