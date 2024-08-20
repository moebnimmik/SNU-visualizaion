import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px

# Streamlit 앱 타이틀
st.title('나만의 월 배당 포트폴리오 구축')

# 포트폴리오를 저장할 리스트
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = []

# 주식 검색 및 주가 정보 가져오기
ticker = st.text_input('주식 티커를 입력하세요 (예: AAPL, MSFT):')
stock_price = None  # 주가를 저장할 변수 초기화

# 주식 추가 기능
if ticker:
    try:
        stock = yf.Ticker(ticker.upper())
        dividends = stock.dividends  # 배당금 데이터 가져오기
        if not dividends.empty:
            st.write(f'{ticker.upper()} 주식 배당금 데이터가 있습니다.')
            # 포트폴리오에 추가
            if ticker.upper() not in [s['ticker'] for s in st.session_state.portfolio]:
                st.session_state.portfolio.append({'ticker': ticker.upper(), 'dividends': dividends})
                st.success(f'{ticker.upper()} 주식이 포트폴리오에 추가되었습니다.')
            else:
                st.warning(f'{ticker.upper()} 주식이 이미 포트폴리오에 있습니다.')
        else:
            st.error(f'{ticker.upper()} 주식에 대한 배당금 데이터가 없습니다.')
    except Exception as e:
        st.error(f"데이터를 가져오는 중 오류 발생: {e}")

# 포트폴리오에 추가된 주식 리스트 보여주기
st.subheader('포트폴리오')
if st.session_state.portfolio:
    # 각 주식의 배당금 정보를 모아서 하나의 DataFrame으로 구성
    portfolio_data = []
    for stock in st.session_state.portfolio:
        dividends = stock['dividends']
        for date, dividend in dividends.items():
            # 월을 1~12로 변환 (날짜의 월만 추출)
            month = pd.to_datetime(date).month
            portfolio_data.append({
                'ticker': stock['ticker'],
                'month': month,
                'dividend': dividend
            })
    
    df = pd.DataFrame(portfolio_data)

    # X축에 12개월을 명시적으로 표시하도록 설정
    months = [str(i) for i in range(1, 13)]  # 1월부터 12월까지
    df['month'] = df['month'].astype(str)  # 월을 문자열로 변환하여 시각화에 사용

    # Plotly로 월별 배당금 누적 막대그래프 그리기
    fig = px.bar(
        df, 
        x='month', 
        y='dividend', 
        color='ticker',
        hover_data=['ticker', 'dividend'],
        labels={'month': '월', 'dividend': '배당금 (₩)'},
        title='포트폴리오의 월별 배당금 흐름',
        barmode='stack',  # 막대를 누적하여 표시
        category_orders={"month": months}  # X축에 1~12까지 표시
    )
    
    st.plotly_chart(fig)

    # 오른쪽에 포트폴리오 목록 표시
    st.sidebar.subheader('포트폴리오 목록')
    tickers_in_portfolio = [stock['ticker'] for stock in st.session_state.portfolio]
    st.sidebar.write(tickers_in_portfolio)

    # 주식 제거 기능 추가
    stock_to_remove = st.sidebar.selectbox('제거할 주식을 선택하세요:', tickers_in_portfolio)
    if st.sidebar.button('제거'):
        st.session_state.portfolio = [stock for stock in st.session_state.portfolio if stock['ticker'] != stock_to_remove]
        st.sidebar.success(f'{stock_to_remove} 주식이 포트폴리오에서 제거되었습니다.')
        st.experimental_rerun()  # 변경사항을 즉시 반영하기 위해 페이지를 다시 로드합니다.
else:
    st.info('포트폴리오에 주식을 추가하세요.')
