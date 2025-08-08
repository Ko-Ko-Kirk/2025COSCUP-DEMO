# OpenAI Agent SDK vs LangGraph 可執行範例

這是 KOKO 在 20250809 的開源人年會裡提供兩個框架的最小可執行範例，展示各自的核心特性。

這場演講是「Source Code Deep Dive into OpenAI Agent SDK vs LangGraph 」，本次演講將深入剖析這兩套框架的原始碼，對比它們在 Agent 核心 Loop、工具呼叫、記憶體管理、狀態轉移與多 Agent 協作等等的面向的架構與設計理念。透過實際程式碼與設計哲學的交叉解析，協助開發者理解這兩種不同 Paradigm 如何各自支撐現代 AI Agent 系統的運作與擴展。

本 Repo 為補充資料。

## 環境設定

### 1. 安裝依賴

使用 uv 管理依賴（更快速的 Python 套件管理器）：

```bash
# 安裝 uv（如果尚未安裝）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 同步依賴並建立虛擬環境
uv sync
```

或使用傳統方式：

```bash
pip install -r requirements.txt
```

### 2. 設定 API Key

在每個範例檔案中，將 `your-api-key-here` 替換為您的 OpenAI API Key


## 範例說明

### OpenAI Agent SDK 範例：客服系統

**檔案**：`openai_agent_example.py`

**特點展示**：
- **Handoff 機制**：分流 Agent 將客戶轉接到專門 Agent
- **工具呼叫**：每個 Agent 有專屬工具
- **串流輸出**：使用 stream_events 即時顯示回應
- **線性執行**：一次一個 Agent 處理

**架構**：
```
客戶查詢
    ↓
分流 Agent（判斷類型）
    ↓ (handoff)
[技術支援 / 帳務專員 / 訂單專員]
    ↓
最終回覆
```

**執行**：
```bash
# 使用 uv
uv run python openai_agent_example.py

# 或傳統方式
python openai_agent_example.py
```

### LangGraph 範例：研究助理系統

**檔案**：`langgraph_example.py`

**特點展示**：
- **Send 並行執行**：同時執行多個搜尋
- **狀態管理**：使用 TypedDict 定義狀態
- **條件路由**：根據狀態決定流程
- **Command 控制**：Human-in-the-Loop 範例
- **多模式串流**：values、updates 等不同串流模式

**架構**：
```
研究問題
    ↓
規劃節點（分解查詢）
    ↓ (conditional_edges 返回 Send)
[搜尋1, 搜尋2, 搜尋3] （並行執行）
    ↓
分析節點（聚合結果）
    ↓
總結節點
    ↓
最終報告
```

**執行**：
```bash
# 使用 uv
uv run python langgraph_example.py

# 或傳統方式
python langgraph_example.py
```

## 主要差異對比

### 1. Agent 協作方式

**OpenAI SDK**：
```python
# Handoff - 線性交接
triage_agent = Agent(
    handoffs=[
        handoff(technical_agent),
        handoff(billing_agent),
    ]
)
```

**LangGraph**：
```python
# Send - 並行派發（透過 conditional_edges）
def route_to_searches(state):
    return [
        Send("search", {"query": q}) 
        for q in state["search_queries"]
    ]

# 在圖中使用
graph.add_conditional_edges("planner", route_to_searches)
```

### 2. 狀態管理

**OpenAI SDK**：
```python
# Context 在 Agent 間傳遞
result = await Runner.run(
    agent,
    input=query,
    context=shared_context
)
```

**LangGraph**：
```python
# TypedDict 定義狀態
class State(TypedDict):
    messages: Annotated[List, operator.add]
    results: str
```

### 3. 串流處理

**OpenAI SDK**：
```python
# 使用 run_streamed 和 stream_events 處理串流
result = Runner.run_streamed(agent, input=query)
async for event in result.stream_events():
    if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
        print(event.data.delta, end="", flush=True)
```

**LangGraph**：
```python
# 多模式串流
async for update in app.astream(
    input,
    stream_mode="updates"  # 或 "values", "messages", "debug"
):
    print(update)
```

## 測試建議

1. **基本測試**：先確認 API Key 設定正確
2. **功能測試**：嘗試不同的查詢測試各個分支
3. **串流測試**：觀察即時串流輸出效果
4. **錯誤處理**：故意輸入無效資料測試錯誤處理

## 常見問題

### Q: Rate Limit 錯誤
A: 在迴圈中加入 `await asyncio.sleep(1)` 避免過快請求

### Q: Import 錯誤
A: 確保已安裝所有依賴：`pip install -r requirements.txt`

### Q: API Key 錯誤
A: 檢查 Key 是否正確，是否有足夠的額度

## 延伸學習

1. **OpenAI SDK**：
   - 加入 Session 保存對話歷史
   - 實作更複雜的 input_filter
   - 加入 guardrails 防護

2. **LangGraph**：
   - 使用 SQLite/PostgreSQL 檢查點
   - 實作子圖（subgraph）
   - 加入更多 Command 控制

## 總結

- **OpenAI Agent SDK**：適合對話式、線性流程的應用
- **LangGraph**：適合複雜工作流、需要並行處理的應用

選擇哪個框架取決於您的具體需求。希望這些範例能幫助您快速上手！
