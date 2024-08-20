import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Streamlit 앱 타이틀
st.title('주식 배당금 및 재투자 분석')

# 사용자 입력 받기
ticker = st.text_input('주식 티커 기호를 입력하세요 (예: AAPL):', 'AAPL')

# 기간 선택기 추가
start_date = st.date_input("시작 날짜", pd.to_datetime("2015-01-01"))
end_date = st.date_input("종료 날짜", pd.to_datetime("2024-01-01"))

if ticker:
    # 데이터 다운로드
    stock = yf.Ticker(ticker)
    stock_data = stock.history(start=start_date, end=end_date, actions=True)

    # 데이터 확인
    if stock_data.empty:
        st.error(f"티커 '{ticker}'에 대한 데이터가 없습니다.")
    else:
        # 배당금 데이터 추출
        if 'Dividends' in stock_data.columns:
            dividends = stock_data['Dividends'].fillna(0)
        else:
            st.warning("이 주식에 대한 배당금 데이터가 없습니다.")
            dividends = pd.Series(index=stock_data.index, data=0)
        
        # 배당금이 있는 데이터만 필터링
        dividend_data = dividends[dividends > 0]

        if dividend_data.empty:
            st.warning("이 주식에 대한 배당금 데이터가 없습니다.")
        else:
            # 주가와 배당금 데이터 시뮬레이션
            initial_shares = 1  # 초기 투자 시 주식 수
            shares = pd.Series(index=stock_data.index, data=np.nan)
            investment_value = pd.Series(index=stock_data.index, data=np.nan)
            
            shares.iloc[0] = initial_shares
            investment_value.iloc[0] = stock_data['Close'].iloc[0] * initial_shares

            for i in range(1, len(stock_data)):
                # 이전 값 가져오기
                prev_shares = shares.iloc[i-1]
                
                # 배당금 재투자
                dividend = dividends.iloc[i]
                if dividend > 0:
                    new_shares = dividend / stock_data['Close'].iloc[i]
                    shares.iloc[i] = prev_shares + new_shares
                else:
                    shares.iloc[i] = prev_shares
                
                # 재투자 후 총 투자 가치
                investment_value.iloc[i] = shares.iloc[i] * stock_data['Close'].iloc[i]

            # 그래프 생성
            fig = go.Figure()

            # 주가 시계열 (종가) - 선 아래 색칠
            fig.add_trace(go.Scatter(
                x=stock_data.index,
                y=stock_data['Close'],
                mode='lines',
                name='종가',
                line=dict(color='blue', width=1),  # 얇은 실선
                fill='tozeroy',  # 선 아래 색칠
                fillcolor='rgba(0, 0, 255, 0.3)',  # 색상과 투명도 설정
                yaxis='y1',
                hovertemplate='주가: %{y:.2f}<extra></extra>'
            ))

            # 배당금 시계열 (막대그래프) - 굵기 조정
            fig.add_trace(go.Bar(
                x=dividend_data.index,
                y=dividend_data,
                name='배당금',
                marker=dict(color='orange'),
                opacity=0.6,  # 투명도 조정
                width=15,  # 막대 너비 조정 (기존 10에서 15로 변경)
                yaxis='y2',
                hovertemplate='배당금: %{y:.2f}<extra></extra>'
            ))

            # 재투자 후 주가 시계열 - 선 아래 색칠 및 색상 조정
            fig.add_trace(go.Scatter(
                x=investment_value.index,
                y=investment_value,
                mode='lines',
                name='재투자 가치',
                line=dict(color='green', width=1),  # 얇은 실선
                fill='tozeroy',  # 선 아래 색칠
                fillcolor='rgba(0, 255, 0, 0.3)',  # 색상과 투명도 설정
                yaxis='y1',
                hovertemplate='재투자 가치: %{y:.2f}<extra></extra>'
            ))

            # 그래프 레이아웃 설정
            fig.update_layout(
                title=f'{ticker} 주가, 배당금 및 재투자 가치',
                xaxis_title='날짜',
                yaxis_title='주가',
                yaxis=dict(
                    title='주가',
                    titlefont=dict(color='black'),
                    tickfont=dict(color='black'),
                    range=[0, stock_data['Close'].max() * 1.5],  # 좌측 y축 범위 설정 (조정)
                    autorange=False,
                    title_standoff=20,  # 제목과 축 사이의 거리
                    tickangle=-45  # x축 레이블 회전
                ),
                yaxis2=dict(
                    title='배당금',
                    titlefont=dict(color='black'),
                    tickfont=dict(color='black'),
                    overlaying='y',
                    side='right',
                    range=[0, 3],  # 오른쪽 y축 범위 설정 (3으로 조정)
                    title_standoff=20  # 제목과 축 사이의 거리
                ),
                hovermode='x unified',
                barmode='overlay',  # 막대그래프와 선그래프 겹치게 표현
                height=800,  # 그래프 높이 설정
                margin=dict(t=50, b=150, l=70, r=70),  # 여백 조정
                xaxis=dict(
                    tickangle=-45,  # x축 레이블 회전
                    tickfont=dict(size=10, color='black')  # x축 레이블 색상 조정
                )
            )

            # 배경과 텍스트 색상 조정
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='lightgray',
                font=dict(size=12, color='black')
            )

            # 그래프에 수익 차이 표시 기능 추가
            fig.update_traces(
                hovertemplate='<b>날짜</b>: %{x}<br>' +
                              '<b>주가</b>: $%{y:.2f}<br>' +
                              '<b>재투자 가치</b>: $%{customdata[0]:.2f}<br>' +
                              '<b>수익 차이</b>: $%{customdata[1]:.2f}<br>' +
                              '<extra></extra>',
                selector=dict(mode='lines+markers')
            )

            # 커스텀 데이터 추가: [재투자 후 가치, 수익 차이]
            fig.update_traces(customdata=np.stack([
                investment_value.values,
                investment_value.values - stock_data['Close'].values
            ], axis=-1))

            # 그래프 보여주기
            st.plotly_chart(fig)