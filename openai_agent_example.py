"""
OpenAI Agent SDK 範例：客服系統
這是一個簡單的客服系統，包含分流、技術支援和帳務處理
"""

import asyncio
import os

# 設定 API Key
api_key = "sk-proj-"  # 請替換為您的 API Key
os.environ["OPENAI_API_KEY"] = api_key

from agents import Agent, Runner, function_tool, handoff
from openai.types.responses import ResponseTextDeltaEvent


# === 工具定義 ===

@function_tool
def check_order_status(order_id: str) -> str:
    """Check order status"""
    # Mock database query
    mock_orders = {
        "12345": "Shipped, expected delivery tomorrow",
        "67890": "Processing, expected to ship in 2 days",
    }
    return mock_orders.get(order_id, f"Order {order_id} not found")


@function_tool
def check_account_balance(account_id: str) -> str:
    """Check account balance"""
    # Mock query
    return f"Account {account_id} balance: $1,234.56"


@function_tool
def process_refund(order_id: str, reason: str) -> str:
    """Process refund request"""
    return f"Refund request for order {order_id} has been received. Reason: {reason}. Expected processing time: 3-5 business days."


@function_tool
def reset_password(email: str) -> str:
    """Reset password"""
    return f"Password reset link has been sent to {email}"


# === Agent 定義 ===

# Technical Support Agent
technical_agent = Agent(
    name="Technical Support",
    instructions="""You are a technical support specialist.
    Help customers with technical issues including:
    - Password resets
    - Account issues
    - System troubleshooting
    Please maintain professionalism and patience.""",
    tools=[reset_password],
    model="gpt-4.1",
)

# Billing Agent
billing_agent = Agent(
    name="Billing Specialist",
    instructions="""You are a billing specialist.
    Handle all money-related issues:
    - Balance inquiries
    - Process refunds
    - Billing issues
    Please ensure accurate handling of financial information.""",
    tools=[check_account_balance, process_refund],
    model="gpt-4.1",
)

# Order Agent
order_agent = Agent(
    name="Order Specialist",
    instructions="""You are an order inquiry specialist.
    Help customers with:
    - Order status inquiries
    - Package tracking
    - Delivery information""",
    tools=[check_order_status],
    model="gpt-4.1",
)

# Triage Agent (Main Entry)
triage_agent = Agent(
    name="Customer Service Triage",
    instructions="""You are the first line of customer service.
    Your task is to understand customer needs and route them to the appropriate specialist:
    - Technical issues → Technical Support
    - Money/billing/refunds → Billing Specialist  
    - Order inquiries → Order Specialist
    
    Briefly understand the issue, then immediately handoff.""",
    handoffs=[
        handoff(technical_agent),
        handoff(billing_agent),
        handoff(order_agent),
    ],
    model="gpt-4.1",
)


# === 主程式 ===

async def handle_customer_query(query: str):
    """處理客戶查詢"""
    print(f"\n{'='*50}")
    print(f"客戶: {query}")
    print(f"{'='*50}\n")
    
    # 執行 Agent
    result = await Runner.run(
        triage_agent,
        input=query,
        # 可選：加入 session 來保存對話歷史
        # session=SQLiteSession("customer_123", "chat.db")
    )
    
    print(f"\n最終回覆: {result.final_output}")
    
    return result


async def stream_example(query: str):
    """串流範例 - 使用正確的 stream_events 方法"""
    print(f"\n{'='*50}")
    print(f"串流模式 - 客戶: {query}")
    print(f"{'='*50}\n")
    
    # 使用 run_streamed 方法（注意不是 run_streaming）
    result = Runner.run_streamed(
        triage_agent,
        input=query
    )
    
    # 串流輸出
    print("回覆: ", end="", flush=True)
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)
    print()  # 最後換行


async def main():
    """主函數"""
    
    # 測試案例
    test_queries = [
        "我的訂單 12345 什麼時候會到？",
        "我忘記密碼了，email 是 user@example.com",
        "我想要退款訂單 67890，因為商品有瑕疵",
        "查詢我的帳戶餘額，帳號是 ACC001",
    ]
    
    # 一般執行
    print("\n" + "="*70)
    print("OpenAI Agent SDK 範例：客服系統")
    print("="*70)
    
    for query in test_queries:
        await handle_customer_query(query)
        await asyncio.sleep(1)  # 避免 rate limit
    
    # 串流執行範例
    print("\n" + "="*70)
    print("串流模式範例")
    print("="*70)
    
    await stream_example("我需要查詢訂單 12345 的狀態")


if __name__ == "__main__":
    # 執行前請確保：
    # 1. pip install openai-agents
    # 2. 設定正確的 OPENAI_API_KEY
    
    asyncio.run(main())