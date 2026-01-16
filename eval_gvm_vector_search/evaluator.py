"""Custom LLM judges for vector search evaluation using MLflow GenAI"""

from mlflow.genai.judges import make_judge


def create_retrieval_relevance_judge(llm_endpoint: str):
    """
    Custom judge for retrieval relevance with insertable rules.
    
    Args:
        llm_endpoint: Databricks LLM endpoint name for judging
        
    Returns:
        MLflow GenAI judge for retrieval relevance
    """
    instructions = """
    你是搜尋/推薦系統的嚴格評審。你要評估「檢索結果是否滿足使用者這個查詢的意圖」，並且重點看 Top 3。

    REQUEST（使用者查詢）:
    {{ inputs }}

    RETRIEVED RESULTS（按排名順序）:
    {{ outputs }}

    評分步驟（請在腦中完成，不要輸出過程）：
    1) 先判斷這個查詢屬於哪種意圖（例如：學習解釋、比較選擇、找清單/範例、找新聞/最新、解決問題/避坑、娛樂消遣）。
    2) 逐一判斷每個結果是否「直接回應該意圖」，不要只看關鍵字重合。
    3) Top 3 權重最高：如果 Top 3 有明顯偏題，即使後面有相關也要大幅扣分。
    4) 覆蓋度：如果使用者意圖需要多面向（例如比較/避坑/步驟/案例），結果是否能互補涵蓋，而不是重複。

    分數定義（1-5）：
    5 = Top 3 幾乎都精準命中意圖，且整體涵蓋互補、缺口很小
    4 = 大多命中意圖，Top 3 只有小瑕疵或有一篇略偏
    3 = 有相關但不穩定：Top 3 有明顯缺口/偏題，或覆蓋面不足
    2 = 多數偏題，只剩少量內容勉強可用
    1 = 幾乎完全不相關

    只輸出一個數字：1 / 2 / 3 / 4 / 5
    """
    
    return make_judge(
        name="retrieval_relevance",
        instructions=instructions,
        model=f"databricks:/{llm_endpoint}"
    )


def create_result_quality_judge(llm_endpoint: str):
    """
    Custom judge for result quality with insertable rules.
    
    Args:
        llm_endpoint: Databricks LLM endpoint name for judging
        
    Returns:
        MLflow GenAI judge for result quality
    """
    instructions = """
    Evaluate the quality and usefulness of the retrieved search results.
    
    REQUEST: {{ inputs }}
    
    RETRIEVED RESULTS:
    {{ outputs }}
    
    CUSTOM QUALITY RULES:
    - Results should be eye-catching and engaging
    - Results should be able induced users to click and read the article based on the query intent
    
    Provide a score from 1-5 where:
    1 = Poor quality, not useful
    2 = Low quality with limited usefulness
    3 = Acceptable quality, moderately useful
    4 = Good quality, very useful
    5 = Excellent quality, highly useful and accurate
    
    Return ONLY the numeric score (1, 2, 3, 4, or 5).
    """
    
    return make_judge(
        name="result_quality",
        instructions=instructions,
        model=f"databricks:/{llm_endpoint}"
    )


def create_ranking_quality_judge(llm_endpoint: str):
    """
    Custom judge for ranking quality with insertable rules.
    
    Args:
        llm_endpoint: Databricks LLM endpoint name for judging
        
    Returns:
        MLflow GenAI judge for ranking quality
    """
    instructions = """
    Evaluate whether the search results are well-ranked.
    
    REQUEST: {{ inputs }}
    
    RETRIEVED RESULTS (in ranking order):
    {{ outputs }}
    
    CUSTOM RANKING RULES:
    - Most relevant results must appear in top positions
    - More engaging and eye-catching results should be ranked higher
    - Ranking should follow logical relevance order
    - Less relevant results should be ranked lower
    
    Provide a score from 1-5 where:
    1 = Poor ranking, relevant results buried
    2 = Weak ranking with some good results lower
    3 = Average ranking, could be improved
    4 = Good ranking, most relevant on top
    5 = Excellent ranking, perfect relevance order
    
    Return ONLY the numeric score (1, 2, 3, 4, or 5).
    """
    
    return make_judge(
        name="ranking_quality",
        instructions=instructions,
        model=f"databricks:/{llm_endpoint}"
    )

