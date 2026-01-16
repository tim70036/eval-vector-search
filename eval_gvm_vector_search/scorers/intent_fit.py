"""Intent Fit scorer for evaluating query intent alignment"""

from eval_gvm_vector_search.scorers.base import BaseScorer


class IntentFitScorer(BaseScorer):
    """
    Evaluates whether retrieved results align with the user's main query intent.
    
    Focuses on Top 3 results with strict intent type matching.
    """
    
    @property
    def name(self) -> str:
        return "intent_fit"
    
    @property
    def description(self) -> str:
        return "Evaluates whether retrieved results align with user's main query intent (focuses on Top 3)"
    
    @property
    def instructions(self) -> str:
        return """1) Intent Fit（意圖吻合度）

你是內容檢索/推薦系統的評審。你的任務是評估「檢索結果是否吻合使用者查詢的主要意圖」，重點看 Top 3。

REQUEST（使用者查詢）:
{{ inputs }}

RETRIEVED RESULTS（按排名順序）:
{{ outputs }}

評估規則（請在腦中完成，不要輸出過程）：
- 先判斷查詢的主要意圖屬於哪類：解釋學習 / 步驟教學 / 比較選擇 / 清單整理 / 避坑排雷 / 查詢定義 / 最新消息 / 深度分析 / 娛樂消遣
- 結果必須「直接服務該意圖」，不能只靠關鍵字擦邊
- Top 3 權重最高：若 Top 3 有偏題或意圖類型錯（例如使用者要比較，結果全是定義科普），要明顯扣分
- 不評估吸睛程度、不評估可信度（那是其他指標）

分數定義（1-5）：
5 = Top 3 幾乎都精準符合主要意圖，且整體一致
4 = 大多符合，Top 3 只有小瑕疵或一篇稍偏
3 = 有相關但意圖不穩，Top 3 存在明顯缺口
2 = 多數不符合主要意圖，只剩少量擦邊內容
1 = 幾乎完全不符合意圖

只輸出一個數字：1 / 2 / 3 / 4 / 5"""