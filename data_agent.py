import pandas as pd, anthropic
from claude_client import get_text, token_cost
from config import API_KEY, MODEL_MAIN

_client = anthropic.Anthropic(api_key=API_KEY)

TOOLS = [
  {
    "name": "describe_dataset",
    "description": "Get shape, columns, dtypes, null counts and basic statistics",
    "input_schema": {
      "type": "object",
      "properties": {
        "include_sample": {
          "type": "boolean",
          "description": "Include first 5 rows"
        }
      }
    }
  },
  {
    "name": "aggregate",
    "description": "Group by a column and compute an aggregation on another column",
    "input_schema": {
      "type": "object",
      "properties": {
        "group_col": {"type": "string", "description": "Column to group by"},
        "value_col": {"type": "string", "description": "Column to aggregate"},
        "func": {
          "type": "string",
          "enum": ["sum","mean","count","max","min","median"],
          "description": "Aggregation function"
        },
        "top_n": {
          "type": "integer",
          "description": "Return only top N results (optional)"
        }
      },
      "required": ["group_col","value_col","func"]
    }
  },
  {
    "name": "detect_outliers",
    "description": "Find statistical outliers in a numeric column using IQR method",
    "input_schema": {
      "type": "object",
      "properties": {
        "column": {"type": "string", "description": "Numeric column to analyse"}
      },
      "required": ["column"]
    }
  },
  {
    "name": "correlation",
    "description": "Calculate correlation between two numeric columns",
    "input_schema": {
      "type": "object",
      "properties": {
        "col_a": {"type": "string"},
        "col_b": {"type": "string"}
      },
      "required": ["col_a","col_b"]
    }
  }
]

def _execute(name: str, inp: dict, df: pd.DataFrame) -> str:
    try:
        if name == "describe_dataset":
            stats  = df.describe(include="all").round(2).to_string()
            nulls  = df.isnull().sum().to_string()
            sample = f"\nSample:\n{df.head(5).to_string()}" if inp.get("include_sample") else ""
            return f"Shape: {df.shape}\nColumns: {list(df.columns)}\nNulls:\n{nulls}\nStats:\n{stats}{sample}"

        elif name == "aggregate":
            g      = df.groupby(inp["group_col"])[inp["value_col"]]
            result = getattr(g, inp["func"])().round(2)
            result = result.sort_values(ascending=False)
            if inp.get("top_n"):
                result = result.head(inp["top_n"])
            return result.to_string()

        elif name == "detect_outliers":
            col     = df[inp["column"]].dropna()
            q1, q3  = col.quantile(0.25), col.quantile(0.75)
            iqr     = q3 - q1
            mask    = (col < q1 - 1.5*iqr) | (col > q3 + 1.5*iqr)
            out     = df[mask]
            return (f"Found {len(out)} outliers in '{inp['column']}'\n"
                    f"Normal range: [{round(q1-1.5*iqr,2)}, {round(q3+1.5*iqr,2)}]\n"
                    f"{out[inp['column']].to_string()}")

        elif name == "correlation":
            r = df[inp["col_a"]].corr(df[inp["col_b"]])
            return (f"Pearson correlation between "
                    f"'{inp['col_a']}' and '{inp['col_b']}': {round(r,4)}")

    except Exception as e:
        return f"Tool error: {str(e)}"
    return "Unknown tool"

def analyse(df: pd.DataFrame, question: str, max_steps: int = 6) -> dict:
    system = """You are a senior data analyst with tool access.
Strategy:
1. Always call describe_dataset first to understand the data.
2. Then use appropriate tools to answer precisely.
3. Give specific numbers in your final answer.
4. Note any data quality issues found."""

    messages   = [{"role": "user", "content": question}]
    tool_log   = []
    total_cost = 0.0

    for step in range(max_steps):
        resp = _client.messages.create(
            model=MODEL_MAIN, max_tokens=2048,
            system=system, messages=messages, tools=TOOLS
        )
        total_cost += token_cost(resp, MODEL_MAIN)["cost_usd"]

        if resp.stop_reason == "end_turn":
            return {
                "answer":   get_text(resp),
                "tool_log": tool_log,
                "steps":    step + 1,
                "cost_usd": round(total_cost, 5)
            }

        # Find all tool_use blocks in this response
        tool_uses = [b for b in resp.content if b.type == "tool_use"]

        if not tool_uses:
            return {"answer": get_text(resp), "tool_log": tool_log,
                    "steps": step, "cost_usd": round(total_cost, 5)}

        # Append the full assistant message (critical — must include all content blocks)
        messages.append({"role": "assistant", "content": resp.content})

        # Execute every tool_use block and collect all results
        tool_results = []
        for tool_block in tool_uses:
            result = _execute(tool_block.name, tool_block.input, df)
            tool_log.append({
                "step":           step + 1,
                "tool":           tool_block.name,
                "input":          tool_block.input,
                "result_preview": result[:300]
            })
            tool_results.append({
                "type":        "tool_result",
                "tool_use_id": tool_block.id,
                "content":     result
            })

        # Append ONE user message containing ALL tool results
        messages.append({"role": "user", "content": tool_results})

    return {"answer": "Max steps reached", "tool_log": tool_log,
            "steps": max_steps, "cost_usd": round(total_cost, 5)}
