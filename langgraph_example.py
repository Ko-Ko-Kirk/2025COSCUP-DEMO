"""
LangGraph 範例：研究助理系統
這是一個多步驟研究系統，包含搜尋、分析、總結等節點
"""

import operator
from typing import TypedDict, Annotated, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END, START
from langgraph.types import Send
from langgraph.checkpoint.memory import MemorySaver

# 設定 API Key
api_key = " sk-proj-"  # 請替換為您的 API Key

# 初始化 LLM
llm = ChatOpenAI(api_key=api_key, model="gpt-4.1", temperature=0.7)


# === 狀態定義 ===

class ResearchState(TypedDict):
    """研究流程的狀態"""
    query: str
    search_queries: List[str]
    search_results: Annotated[List[str], operator.add]  # 累加搜尋結果
    analysis: str
    summary: str
    messages: Annotated[List[BaseMessage], operator.add]  # 累加訊息


# === 節點函數 ===

def planner_node(state: ResearchState) -> ResearchState:
    """規劃節點：將問題分解為搜尋查詢"""
    query = state["query"]
    
    prompt = f"""將以下研究問題分解為 2-3 個具體的搜尋查詢：
    問題：{query}
    
    請返回用逗號分隔的查詢列表。"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    queries = [q.strip() for q in response.content.split(",")]
    
    return {
        "search_queries": queries,
        "messages": [AIMessage(content=f"已規劃搜尋查詢：{', '.join(queries)}")]
    }


def search_node(state: dict) -> dict:
    """搜尋節點：執行單個搜尋"""
    # 這個節點會被 Send 多次調用，每次處理不同的查詢
    query = state.get("query", "")
    
    # 模擬搜尋結果
    mock_results = {
        "量子計算": "量子計算使用量子位元（qubit）進行運算，可以同時處於 0 和 1 的疊加態。",
        "機器學習": "機器學習是 AI 的子領域，讓電腦從資料中學習模式而無需明確編程。",
        "區塊鏈": "區塊鏈是分散式帳本技術，透過加密確保資料不可竄改。",
        "雲端運算": "雲端運算提供按需的運算資源，包括儲存、處理能力和應用程式。",
        "物聯網": "物聯網（IoT）連接實體設備到網際網路，實現資料收集和遠端控制。",
    }
    
    # 簡單的關鍵字匹配
    result = "未找到相關資訊"
    for key, value in mock_results.items():
        if key in query:
            result = value
            break
    
    return {
        "search_results": [f"查詢「{query}」：{result}"]
    }


def route_to_searches(state: ResearchState):
    """路由函數：使用 Send 並行搜尋 (用於 conditional_edges)"""
    queries = state.get("search_queries", [])
    
    # 為每個查詢創建 Send 物件
    sends = []
    for query in queries:
        sends.append(Send("search", {"query": query}))
    
    return sends


def analyzer_node(state: ResearchState) -> ResearchState:
    """分析節點：分析搜尋結果"""
    results = state.get("search_results", [])
    query = state["query"]
    
    if not results:
        return {"analysis": "沒有找到相關資訊"}
    
    prompt = f"""基於以下搜尋結果，分析問題「{query}」：

搜尋結果：
{chr(10).join(results)}

請提供深入的分析。"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "analysis": response.content,
        "messages": [AIMessage(content=f"分析完成：{response.content[:100]}...")]
    }


def summary_node(state: ResearchState) -> ResearchState:
    """總結節點：生成最終報告"""
    analysis = state.get("analysis", "")
    query = state["query"]
    
    prompt = f"""為問題「{query}」生成簡潔的研究報告：

分析內容：
{analysis}

請用 3-5 句話總結關鍵發現。"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "summary": response.content,
        "messages": [AIMessage(content=f"最終報告：{response.content}")]
    }


def should_continue(state: ResearchState) -> str:
    """條件函數：決定是否繼續"""
    if state.get("search_results"):
        return "continue"
    else:
        return "end"


# === 構建圖 ===

def build_research_graph():
    """構建研究助理圖"""
    graph = StateGraph(ResearchState)
    
    # 添加節點
    graph.add_node("planner", planner_node)
    graph.add_node("search", search_node)
    graph.add_node("analyzer", analyzer_node)
    graph.add_node("summary", summary_node)
    
    # 定義邊
    graph.add_edge(START, "planner")
    
    # planner 使用條件邊返回 Send 物件進行並行搜尋
    graph.add_conditional_edges(
        "planner",
        route_to_searches,  # 返回 Send 列表的路由函數
    )
    
    # search 完成後到 analyzer
    graph.add_edge("search", "analyzer")
    
    # analyzer 後的條件判斷
    graph.add_conditional_edges(
        "analyzer",
        should_continue,
        {
            "continue": "summary",
            "end": END
        }
    )
    
    graph.add_edge("summary", END)
    
    # 編譯圖（可選：加入檢查點）
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# === 主程式 ===

async def run_research(query: str):
    """執行研究"""
    print(f"\n{'='*50}")
    print(f"研究問題: {query}")
    print(f"{'='*50}\n")
    
    # 建立圖
    app = build_research_graph()
    
    # 配置（用於檢查點）
    config = {"configurable": {"thread_id": "research_001"}}
    
    # 執行
    result = await app.ainvoke(
        {"query": query},
        config=config
    )
    
    print("\n=== 研究結果 ===")
    print(f"搜尋查詢: {result.get('search_queries', [])}")
    print(f"\n搜尋結果:")
    for r in result.get("search_results", []):
        print(f"  - {r}")
    print(f"\n分析: {result.get('analysis', 'N/A')}")
    print(f"\n總結: {result.get('summary', 'N/A')}")
    
    return result


async def stream_research(query: str):
    """串流執行研究"""
    print(f"\n{'='*50}")
    print(f"串流模式 - 研究問題: {query}")
    print(f"{'='*50}\n")
    
    app = build_research_graph()
    config = {"configurable": {"thread_id": "stream_001"}}
    
    # 串流不同模式
    print("\n=== 串流更新 (updates) ===")
    async for chunk in app.astream(
        {"query": query},
        config=config,
        stream_mode="updates"
    ):
        # chunk 是一個元組 (node_name, update_data)
        if isinstance(chunk, tuple) and len(chunk) == 2:
            node_name, update = chunk
            print(f"節點 '{node_name}' 輸出更新")
            if isinstance(update, dict) and "messages" in update:
                for msg in update["messages"]:
                    print(f"  → {msg.content[:100]}")
        else:
            # 或者 chunk 可能是字典格式
            for node_name, update in chunk.items():
                print(f"節點 '{node_name}' 輸出更新")
                if isinstance(update, dict) and "messages" in update:
                    for msg in update["messages"]:
                        print(f"  → {msg.content[:100]}")
    
    print("\n=== 串流值 (values) - 顯示完整狀態 ===")
    final_state = None
    async for state in app.astream(
        {"query": query},
        config={"configurable": {"thread_id": "stream_002"}},
        stream_mode="values"
    ):
        final_state = state
        if "messages" in state:
            print(f"訊息數: {len(state['messages'])}")
    
    return final_state


async def demonstrate_command():
    """展示 Command 使用（Human-in-the-Loop）"""
    from langgraph.types import Command, interrupt
    
    class ApprovalState(TypedDict):
        task: str
        approved: bool
        result: str
    
    def review_node(state: ApprovalState):
        """需要人工審核的節點"""
        if not state.get("approved", False):
            # 中斷執行，等待審核
            interrupt("需要人工審核")
        
        # 審核通過後繼續
        return {"result": f"已批准執行：{state['task']}"}
    
    def execute_node(state: ApprovalState):
        """執行節點"""
        return {"result": f"任務 '{state['task']}' 執行完成"}
    
    # 構建需要審核的圖
    approval_graph = StateGraph(ApprovalState)
    approval_graph.add_node("review", review_node)
    approval_graph.add_node("execute", execute_node)
    
    approval_graph.add_edge(START, "review")
    approval_graph.add_edge("review", "execute")
    approval_graph.add_edge("execute", END)
    
    app = approval_graph.compile(checkpointer=MemorySaver())
    
    print("\n=== Command 範例：Human-in-the-Loop ===")
    
    # 第一次執行（會中斷）
    config = {"configurable": {"thread_id": "approval_001"}}
    try:
        result = await app.ainvoke(
            {"task": "刪除資料庫"},
            config=config
        )
    except Exception as e:
        print(f"執行中斷：{e}")
    
    # 模擬人工審核後恢復
    print("\n模擬人工審核通過...")
    
    # 使用 Command 恢復執行
    result = await app.ainvoke(
        Command(
            update={"approved": True},
            resume={"approval": "已通過審核"}
        ),
        config=config
    )
    
    print(f"最終結果：{result.get('result')}")


async def main():
    """主函數"""
    
    print("\n" + "="*70)
    print("LangGraph 範例：研究助理系統")
    print("="*70)
    
    # 測試問題
    test_queries = [
        "什麼是量子計算？",
        "機器學習和深度學習的區別",
        "區塊鏈技術的應用",
    ]
    
    # 一般執行
    for query in test_queries[:1]:  # 只測試第一個
        await run_research(query)
    
    # 串流執行
    print("\n" + "="*70)
    print("串流模式範例")
    print("="*70)
    
    await stream_research("雲端運算的優勢")
    
    # Command 範例
    print("\n" + "="*70)
    print("Command 控制範例")
    print("="*70)
    
    await demonstrate_command()


if __name__ == "__main__":
    # 執行前請確保：
    # 1. pip install langgraph langchain-openai
    # 2. 設定正確的 OPENAI_API_KEY
    
    import asyncio
    asyncio.run(main())