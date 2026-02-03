from typing import Dict, List

from fastmcp import FastMCP

mcp = FastMCP(name="ArxivExplorer")

print("✅ ArxivExplorer server initialized.")


# --- Dynamic Resource: Suggested AI research topics ---
@mcp.resource("resource://ai/arxiv_topics")
def arxiv_topics() -> List[str]:
    return [
        "Transformer interpretability",
        "Efficient large-scale model training",
        "Federated learning privacy",
        "Neural network pruning",
    ]


print("✅ Resource 'resource://ai/arxiv_topics' registered.")

# --- Tool: Search ArXiv for recent papers ---
@mcp.tool(annotations={"title": "Search Arxiv"})
def search_arxiv(query: str, max_results: int = 5) -> List[Dict]:
    """
    Queries ArXiv via Tavily, returning title + link for each paper,
    and *only* ArXiv results.
    """
    #resp = tavily.search(
    #    query=f"site:arxiv.org {query}",
    #    max_results=max_results,
    #)

    return [
        {"title": r["title"].strip(), "url": r["url"]}
        for r in [{"title": "Sample Paper", "url": "https://arxiv.org/abs/1234.5678"}][:max_results]
    ]


# --- Tool: Summarize an ArXiv paper ---
@mcp.tool(annotations={"title": "Summarize Paper"})
def summarize_paper(paper_url: str) -> str:
    """
    Returns a one-paragraph summary of the paper at the given URL.
    """
    prompt = f"Summarize the key contributions of this ArXiv paper: {paper_url}"
    return prompt
    #return tavily.qna_search(query=prompt)


print("✅ Tools 'Search Arxiv' and 'Summarize Paper' registered.")


# --- Prompt Template: Explore a topic thoroughly ---
@mcp.prompt
def explore_topic_prompt(topic: str) -> str:
    return (
        f"I want to explore recent work on '{topic}'.\n"
        f"1. Call the 'Search Arxiv' tool to find the 5 most recent papers.\n"
        f"2. For each paper URL, call 'Summarize Paper' to extract its key contributions.\n"
        f"3. Combine all summaries into an overview report."
    )


print("✅ Prompt 'explore_topic_prompt' registered.")


if __name__ == "__main__":
    print("\n🚀 Starting ArxivExplorer Server...")
    mcp.settings.port = 8002
    mcp.run(transport="streamable-http")