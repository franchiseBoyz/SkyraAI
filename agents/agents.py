import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.planning import SequentialPlanner
from semantic_kernel.skill_definition import sk_function, sk_function_context_parameter
import yfinance as yf
import numpy as np
import pandas as pd
from ta import add_all_ta_features
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import MACD
from typing import Dict, Optional
import asyncio
from datetime import datetime, timedelta

# Initialize Semantic Kernel
kernel = sk.Kernel()
api_key = "your-openai-api-key"
kernel.add_chat_service("gpt-4", OpenAIChatCompletion("gpt-4", api_key))

# ======================
# WARREN BUFFETT AGENT
# ======================
class BuffettOracleSkill:
    def __init__(self):
        self.moat_threshold = 0.7
        self.management_threshold = 0.8
        self.profitability_threshold = 0.8
        self.debt_threshold = 0.5  # Debt/Equity ratio
        self.margin_of_safety_threshold = 0.3  # 30%

    @sk_function(
        description="Analyze a company using Warren Buffett's principles of moat, management, and margin of safety",
        name="analyze_buffett_style"
    )
    @sk_function_context_parameter(name="ticker", description="Stock ticker symbol")
    async def analyze_buffett_style(self, context: sk.SKContext) -> str:
        ticker = context["ticker"]
        stock = yf.Ticker(ticker)
        
        try:
            # Get financial data
            financials = stock.financials
            balance_sheet = stock.balance_sheet
            cash_flow = stock.cashflow
            current_price = stock.history(period="1d")['Close'].iloc[-1]

            # Calculate Buffett's metrics
            analysis = {
                "economic_moat": self._calculate_moat(stock, financials),
                "management_quality": self._evaluate_management(stock),
                "profitability": self._analyze_profitability(financials),
                "debt_levels": self._check_debt(balance_sheet),
                "owner_earnings": self._calculate_owner_earnings(cash_flow, stock.info.get('sharesOutstanding', 1)),
                "margin_of_safety": self._calculate_margin_of_safety(stock, current_price),
                "current_price": current_price,
                "business_summary": stock.info.get('longBusinessSummary', '')
            }

            # Determine if it's a "wonderful company"
            analysis['wonderful_company'] = all([
                analysis['economic_moat'] >= self.moat_threshold,
                analysis['management_quality'] >= self.management_threshold,
                analysis['profitability'] >= self.profitability_threshold,
                analysis['debt_levels'] <= self.debt_threshold
            ])

            # Generate Buffett-style analysis
            prompt = f"""
            **Business Summary**:
            {analysis['business_summary']}

            **Warren Buffett Analysis for {ticker}** (Current Price: ${analysis['current_price']:.2f}):

            1. ECONOMIC MOAT (Scale: 0-1): {analysis['economic_moat']:.2f} 
            - {'Strong moat (>0.7)' if analysis['economic_moat'] >= 0.7 else '⚠Weak moat'}
            - Measures durable competitive advantage through brand, cost advantages, and switching costs

            2. MANAGEMENT QUALITY (Scale: 0-1): {analysis['management_quality']:.2f}
            - {'Excellent (>0.8)' if analysis['management_quality'] >= 0.8 else ' Needs improvement'}
            - Evaluates capital allocation and shareholder orientation

            3. PROFITABILITY (Scale: 0-1): {analysis['profitability']:.2f}
            - {'Excellent (>0.8)' if analysis['profitability'] >= 0.8 else '⚠Needs improvement'}
            - Consistent high returns on capital

            4. DEBT LEVELS (Debt/Equity): {analysis['debt_levels']:.2f}
            - {'Conservative (<0.5)' if analysis['debt_levels'] <= 0.5 else ' Too leveraged'}

            5. OWNER EARNINGS: ${analysis['owner_earnings']:,.2f} per share
            - Buffett's preferred measure of true cash generation

            6. MARGIN OF SAFETY: {analysis['margin_of_safety']:.2%}
            - {' Sufficient (>30%)' if analysis['margin_of_safety'] >= 0.3 else ' Insufficient'}

            FINAL VERDICT: {'Wonderful Company at Fair Price' if analysis['wonderful_company'] and analysis['margin_of_safety'] >= 0.3 else '⚠️ Does Not Meet Buffett Criteria'}

            Provide detailed analysis in Buffett's style covering:
            - Business quality assessment
            - Management evaluation
            - Valuation attractiveness
            - Key risks
            - Long-term prospects
            """

            buffett_analysis = kernel.create_semantic_function(prompt, max_tokens=800)
            return await buffett_analysis.invoke_async()
            
        except Exception as e:
            return f"Buffett analysis failed: {str(e)}"

    def _calculate_moat(self, stock, financials) -> float:
        """Calculate economic moat score (0-1)"""
        try:
            # 1. Brand strength (simplified)
            brand_score = 0.7 if any(x in stock.info.get('longBusinessSummary','').lower() 
                                 for x in ['brand', 'patent', 'unique']) else 0.5
            
            # 2. Gross margin stability (last 3 years)
            gross_margin = financials.loc['Gross Profit'] / financials.loc['Total Revenue']
            margin_stability = 1 - (np.std(gross_margin[-3:]) / np.mean(gross_margin[-3:]))
            
            # 3. ROIC consistency (last 3 years)
            try:
                roic = financials.loc['Net Income'][-3:] / (
                    balance_sheet.loc['Total Assets'][-3:] - 
                    balance_sheet.loc['Total Current Liabilities'][-3:])
                roic_consistency = 1 - (np.std(roic) / np.mean(roic))
            except:
                roic_consistency = 0.5
                
            return float(0.4*brand_score + 0.4*margin_stability + 0.2*roic_consistency)
        except:
            return 0.5

    def _evaluate_management(self, stock) -> float:
        """Evaluate management quality (0-1)"""
        try:
            # 1. ROE consistency (3 years)
            roe = stock.financials.loc['Net Income'][-3:] / stock.balance_sheet.loc['Total Stockholder Equity'][-3:]
            roe_consistency = 1 - (np.std(roe) / np.mean(roe))
            
            # 2. Capital allocation (buybacks/dividends)
            try:
                actions = stock.actions
                dividend_score = 0.2 if not actions.empty and 'Dividends' in actions else 0
                shares = stock.get_shares_full(start=datetime.now()-timedelta(days=365*3), end=datetime.now())
                buyback_score = 0.1 if shares.iloc[-1] < shares.iloc[0] else 0
            except:
                dividend_score = buyback_score = 0
            
            # 3. Insider ownership
            insider_score = 0.1 if stock.info.get('heldPercentInsiders',0) > 0.05 else 0
            
            return min(1.0, 0.6*roe_consistency + dividend_score + buyback_score + insider_score)
        except:
            return 0.5

    def _analyze_profitability(self, financials) -> float:
        """Analyze consistent profitability (0-1)"""
        try:
            # 1. Gross margin (3Y avg)
            gross_margin = np.mean((financials.loc['Gross Profit'] / financials.loc['Total Revenue'])[-3:])
            
            # 2. Net margin consistency
            net_margin = financials.loc['Net Income'] / financials.loc['Total Revenue']
            margin_consistency = 1 - (np.std(net_margin[-3:]) / np.mean(net_margin[-3:]))
            
            # 3. Positive earnings years
            positive_earnings = sum(financials.loc['Net Income'] > 0) / len(financials.columns)
            
            return float(0.5*gross_margin + 0.3*margin_consistency + 0.2*positive_earnings)
        except:
            return 0.5
        
        

    def _check_debt(self, balance_sheet) -> float:
        """Calculate Debt/Equity ratio"""
        try:
            total_debt = balance_sheet.loc['Total Liab'][0]
            total_equity = balance_sheet.loc['Total Stockholder Equity'][0]
            return float(total_debt / total_equity)
        except:
            return 1.0  # Conservative default

    def _calculate_owner_earnings(self, cash_flow, shares_outstanding) -> float:
        """Calculate Buffett's owner earnings (Net Income + Depreciation - CapEx)"""
        try:
            owner_earnings = (
                cash_flow.loc['Net Income'][0] + 
                cash_flow.loc['Depreciation'][0] - 
                cash_flow.loc['Capital Expenditure'][0]
            )
            return owner_earnings / shares_outstanding
        except:
            return 0.0

    def _calculate_margin_of_safety(self, stock, current_price) -> float:
        """Calculate margin of safety using Graham's formula"""
        try:
            eps = stock.info.get('trailingEps', 0)
            book_value = stock.info.get('bookValue', 1)
            graham_value = np.sqrt(22.5 * eps * book_value)
            return (graham_value - current_price) / graham_value
        except:
            return 0.0

# ======================
# FUNDAMENTALS AGENT
# ======================
class FundamentalsAnalysisSkill:
    @sk_function(
        description="Analyze company fundamentals using financial statements",
        name="analyze_fundamentals"
    )
    @sk_function_context_parameter(name="ticker", description="Stock ticker symbol")
    async def analyze_fundamentals(self, context: sk.SKContext) -> str:
        ticker = context["ticker"]
        stock = yf.Ticker(ticker)
        
        try:
            # Get financial data
            financials = stock.financials
            balance_sheet = stock.balance_sheet
            cash_flow = stock.cashflow
            
            # Calculate key metrics
            metrics = {
                "revenue_growth_3y": self._calculate_growth(financials.loc['Total Revenue'][-3:]),
                "eps_growth_3y": self._calculate_growth(financials.loc['Net Income'][-3:]),
                "roe": self._calculate_roe(financials, balance_sheet),
                "roic": self._calculate_roic(financials, balance_sheet),
                "current_ratio": self._calculate_current_ratio(balance_sheet),
                "free_cash_flow": cash_flow.loc['Free Cash Flow'].iloc[0] / 1e6,  # in millions
                "operating_margin": (financials.loc['Operating Income'] / financials.loc['Total Revenue']).iloc[0],
                "pe_ratio": stock.info.get('trailingPE', 0),
                "peg_ratio": stock.info.get('pegRatio', 0),
                "dividend_yield": stock.info.get('dividendYield', 0) * 100 if stock.info.get('dividendYield') else 0
            }
            
            # Generate analysis using LLM
            prompt = f"""
            **Fundamental Analysis for {ticker}**:
            
            Key Metrics:
            - Revenue Growth (3Y): {metrics['revenue_growth_3y']:.2%}
            - EPS Growth (3Y): {metrics['eps_growth_3y']:.2%}
            - ROE: {metrics['roe']:.2%} 
            - ROIC: {metrics['roic']:.2%}
            - Current Ratio: {metrics['current_ratio']:.2f}
            - Free Cash Flow: ${metrics['free_cash_flow']:,.2f}M
            - Operating Margin: {metrics['operating_margin']:.2%}
            - P/E Ratio: {metrics['pe_ratio']:.2f}
            - PEG Ratio: {metrics['peg_ratio']:.2f}
            - Dividend Yield: {metrics['dividend_yield']:.2f}%
            
            Provide detailed analysis covering:
            1. Growth Prospects
            2. Profitability Quality
            3. Financial Health
            4. Valuation Context
            5. Competitive Position
            """

            fundamental_analysis = kernel.create_semantic_function(prompt, max_tokens=600)
            return await fundamental_analysis.invoke_async()
            
        except Exception as e:
            return f"Fundamental analysis failed: {str(e)}"

    def _calculate_growth(self, series: pd.Series) -> float:
        return float((series.iloc[0] - series.iloc[-1]) / abs(series.iloc[-1]))

    def _calculate_roe(self, financials, balance_sheet) -> float:
        return float(financials.loc['Net Income'].iloc[0] / balance_sheet.loc['Total Stockholder Equity'].iloc[0])

    def _calculate_roic(self, financials, balance_sheet) -> float:
        return float(financials.loc['Net Income'].iloc[0] / 
                    (balance_sheet.loc['Total Assets'].iloc[0] - 
                     balance_sheet.loc['Total Current Liabilities'].iloc[0]))

    def _calculate_current_ratio(self, balance_sheet) -> float:
        return float(balance_sheet.loc['Total Current Assets'].iloc[0] / 
                    balance_sheet.loc['Total Current Liabilities'].iloc[0])

# ======================
# TECHNICALS AGENT
# ======================
class TechnicalAnalysisSkill:
    @sk_function(
        description="Perform technical analysis on stock price data",
        name="analyze_technicals"
    )
    @sk_function_context_parameter(name="ticker", description="Stock ticker symbol")
    @sk_function_context_parameter(name="period", description="Lookback period", default_value="1y")
    async def analyze_technicals(self, context: sk.SKContext) -> str:
        ticker = context["ticker"]
        period = context["period"]
        
        try:
            # Get price data
            data = yf.download(ticker, period=period)
            if data.empty:
                return "No price data available"
                
            # Add technical indicators
            data = add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume")
            
            # Calculate signals
            signals = {
                "rsi": RSIIndicator(data['Close']).rsi().iloc[-1],
                "macd": MACD(data['Close']).macd_diff().iloc[-1],
                "50_day_ma": data['Close'].rolling(50).mean().iloc[-1],
                "200_day_ma": data['Close'].rolling(200).mean().iloc[-1],
                "current_price": data['Close'].iloc[-1],
                "bb_upper": BollingerBands(data['Close']).bollinger_hband().iloc[-1],
                "bb_lower": BollingerBands(data['Close']).bollinger_lband().iloc[-1],
                "volume_avg": data['Volume'].rolling(20).mean().iloc[-1],
                "volume_current": data['Volume'].iloc[-1]
            }
            
            # Generate technical analysis
            prompt = f"""
            **Technical Analysis for {ticker}** (Current Price: ${signals['current_price']:.2f}):
            
            Key Indicators:
            - RSI (14-day): {signals['rsi']:.2f} {'(Oversold <30)' if signals['rsi'] < 30 else '(Overbought >70)' if signals['rsi'] > 70 else ''}
            - MACD: {signals['macd']:.2f} {'(Bullish >0)' if signals['macd'] > 0 else '(Bearish <0)'}
            - 50-Day MA: ${signals['50_day_ma']:.2f}
            - 200-Day MA: ${signals['200_day_ma']:.2f}
            - Golden Cross: {'Yes' if signals['50_day_ma'] > signals['200_day_ma'] else '❌ No'}
            - Bollinger Bands: Upper ${signals['bb_upper']:.2f} | Lower ${signals['bb_lower']:.2f}
            - Volume: Current {signals['volume_current']/1e6:.2f}M vs Avg {signals['volume_avg']/1e6:.2f}M
            
            Provide detailed analysis covering:
            1. Current Trend Identification
            2. Support/Resistance Levels  
            3. Momentum Indicators
            4. Volume Analysis
            5. Recommended Entry/Exit Points
            6. Key Chart Patterns
            """

            technical_analysis = kernel.create_semantic_function(prompt, max_tokens=600)
            return await technical_analysis.invoke_async()
            
        except Exception as e:
            return f"Technical analysis failed: {str(e)}"

# ======================
# SENTIMENT AGENT
# ======================
class SentimentAnalysisSkill:
    @sk_function(
        description="Analyze market sentiment from news and social media",
        name="analyze_sentiment"
    )
    @sk_function_context_parameter(name="ticker", description="Stock ticker symbol")
    async def analyze_sentiment(self, context: sk.SKContext) -> str:
        ticker = context["ticker"]
        stock = yf.Ticker(ticker)
        
        try:
            # Get sentiment indicators (simulated - would use APIs in production)
            indicators = {
                "analyst_recommendation": stock.info.get('recommendationMean', 'Hold'),
                "short_interest": stock.info.get('shortPercentOfFloat', 0)*100,
                "institutional_ownership": stock.info.get('heldPercentInstitutions', 0)*100,
                "news_sentiment": np.random.uniform(-1, 1),  # Simulated
                "social_sentiment": np.random.uniform(-1, 1)  # Simulated
            }
            
            # Generate sentiment analysis
            prompt = f"""
            **Market Sentiment Analysis for {ticker}**:
            
            Key Indicators:
            - Analyst Consensus: {indicators['analyst_recommendation'].title()}
            - Short Interest: {indicators['short_interest']:.2f}% of float
            - Institutional Ownership: {indicators['institutional_ownership']:.2f}%
            - News Sentiment: {'Bullish' if indicators['news_sentiment'] > 0.3 else 'Bearish' if indicators['news_sentiment'] < -0.3 else 'Neutral'}
            - Social Media Sentiment: {'Bullish' if indicators['social_sentiment'] > 0.3 else 'Bearish' if indicators['social_sentiment'] < -0.3 else 'Neutral'}
            
            Provide detailed analysis covering:
            1. Analyst Community View
            2. Short Interest Implications  
            3. Institutional Activity
            4. News Flow Analysis
            5. Social Media Buzz
            6. Contrarian Indicators
            """

            sentiment_analysis = kernel.create_semantic_function(prompt, max_tokens=500)
            return await sentiment_analysis.invoke_async()
            
        except Exception as e:
            return f"Sentiment analysis failed: {str(e)}"

# ======================
# VALUATION AGENT
# ======================
class ValuationSkill:
    @sk_function(
        description="Calculate intrinsic value using multiple models",
        name="calculate_valuation"
    )
    @sk_function_context_parameter(name="ticker", description="Stock ticker symbol")
    async def calculate_valuation(self, context: sk.SKContext) -> str:
        ticker = context["ticker"]
        stock = yf.Ticker(ticker)
        
        try:
            # Get current price
            current_price = stock.history(period="1d")['Close'].iloc[-1]
            
            # Calculate multiple valuation metrics
            valuations = {
                "dcf": self._dcf_valuation(stock),
                "graham": self._graham_formula(stock),
                "earnings_power": self._earnings_power_value(stock),
                "ev_ebitda": stock.info.get('enterpriseToEbitda', 0),
                "price_to_book": stock.info.get('priceToBook', 0),
                "price_to_sales": stock.info.get('priceToSales', 0),
                "dividend_yield": stock.info.get('dividendYield', 0)*100 if stock.info.get('dividendYield') else 0,
                "historical_pe": self._historical_pe(stock)
            }
            
            # Generate valuation analysis
            prompt = f"""
            **Valuation Analysis for {ticker}** (Current Price: ${current_price:.2f}):
            
            Valuation Models:
            - DCF Valuation: ${valuations['dcf']:.2f} ({'Undervalued' if valuations['dcf'] > current_price else 'Overvalued'})
            - Graham Formula: ${valuations['graham']:.2f} 
            - Earnings Power Value: ${valuations['earnings_power']:.2f}
            
            Relative Valuation:
            - EV/EBITDA: {valuations['ev_ebitda']:.1f}x
            - P/B Ratio: {valuations['price_to_book']:.1f}x  
            - P/S Ratio: {valuations['price_to_sales']:.1f}x
            - Dividend Yield: {valuations['dividend_yield']:.2f}%
            - Historical P/E: {valuations['historical_pe']:.1f}x vs Current {stock.info.get('trailingPE',0):.1f}x
            
            Provide detailed analysis covering:
            1. Margin of Safety Calculation
            2. Most Appropriate Valuation Method  
            3. Relative vs Historical Valuation
            4. Industry Comparison
            5. Growth vs Value Perspective
            """

            valuation_analysis = kernel.create_semantic_function(prompt, max_tokens=600)
            return await valuation_analysis.invoke_async()
            
        except Exception as e:
            return f"Valuation failed: {str(e)}"

    def _dcf_valuation(self, stock, growth_rate=0.03, discount_rate=0.08) -> float:
        """Discounted Cash Flow valuation"""
        try:
            cash_flows = stock.cashflow.loc['Free Cash Flow']
            terminal_value = cash_flows.iloc[0] * (1 + growth_rate) / (discount_rate - growth_rate)
            present_value = terminal_value / (1 + discount_rate)**5
            return present_value / stock.info['sharesOutstanding']
        except:
            return 0.0

    def _graham_formula(self, stock) -> float:
        """Benjamin Graham's valuation formula"""
        try:
            eps = stock.info.get('trailingEps', 0)
            book_value = stock.info.get('bookValue', 1)
            return np.sqrt(22.5 * eps * book_value)
        except:
            return 0.0

    def _earnings_power_value(self, stock) -> float:
        """Earnings Power Value (EPV)"""
        try:
            avg_earnings = np.mean(stock.financials.loc['Net Income'][-3:])
            return avg_earnings / (0.08 * stock.info['sharesOutstanding'])  # 8% discount rate
        except:
            return 0.0

    def _historical_pe(self, stock, years=5) -> float:
        """Calculate average historical P/E ratio"""
        try:
            history = stock.history(period=f"{years}y")
            eps = stock.info.get('trailingEps', 1)
            return np.mean(history['Close'] / eps)
        except:
            return 0.0

# ======================
# PORTFOLIO MANAGER
# ======================
class PortfolioManagementSkill:
    @sk_function(
        description="Make final investment decision synthesizing all analyses",
        name="make_decision"
    )
    @sk_function_context_parameter(name="ticker", description="Stock ticker symbol")
    async def make_decision(self, context: sk.SKContext) -> str:
        ticker = context["ticker"]
        
        # Execute all analyses in parallel
        tasks = [
            kernel.skills.get_function("BuffettOracleSkill", "analyze_buffett_style").invoke_async(context),
            kernel.skills.get_function("FundamentalsAnalysisSkill", "analyze_fundamentals").invoke_async(context),
            kernel.skills.get_function("TechnicalAnalysisSkill", "analyze_technicals").invoke_async(context),
            kernel.skills.get_function("SentimentAnalysisSkill", "analyze_sentiment").invoke_async(context),
            kernel.skills.get_function("ValuationSkill", "calculate_valuation").invoke_async(context)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Generate final decision
        prompt = f"""
        **Final Investment Decision for {ticker}**
        
        Synthesize these analyses into a comprehensive recommendation:
        
        1. WARREN BUFFETT ANALYSIS:
        {results[0]}
        
        2. FUNDAMENTAL ANALYSIS:
        {results[1]}
        
        3. TECHNICAL ANALYSIS:
        {results[2]}
        
        4. MARKET SENTIMENT:
        {results[3]}
        
        5. VALUATION ANALYSIS:
        {results[4]}
        
        Provide:
        - Final Recommendation (BUY/HOLD/SELL) with Confidence Level
        - Price Target Range with Time Horizon
        - Suggested Position Size (% of portfolio)
        - Key Risks to Monitor
        - Ideal Entry Points
        - Catalyst Watchlist
        """

        decision_analysis = kernel.create_semantic_function(prompt, max_tokens=800)
        return await decision_analysis.invoke_async()

# ======================
# INITIALIZE AND RUN
# ======================
async def main():
    # Register all skills
    kernel.import_skill(BuffettOracleSkill(), "BuffettOracleSkill")
    kernel.import_skill(FundamentalsAnalysisSkill(), "FundamentalsAnalysisSkill")
    kernel.import_skill(TechnicalAnalysisSkill(), "TechnicalAnalysisSkill")
    kernel.import_skill(SentimentAnalysisSkill(), "SentimentAnalysisSkill")
    kernel.import_skill(ValuationSkill(), "ValuationSkill")
    kernel.import_skill(PortfolioManagementSkill(), "PortfolioManagementSkill")
    
    # Analyze a stock
    ticker = "AAPL"
    context = kernel.create_new_context()
    context["ticker"] = ticker
    
    portfolio_manager = kernel.skills.get_function("PortfolioManagementSkill", "make_decision")
    result = await portfolio_manager.invoke_async(context=context)
    
    print(f"\n=== COMPLETE ANALYSIS FOR {ticker} ===")
    print(result.result)

if __name__ == "__main__":
    asyncio.run(main())
