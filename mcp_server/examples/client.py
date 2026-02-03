import ast
import asyncio
import pprint

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport


# --- Configuration ---
SERVER_URL = "http://localhost:8000/mcp"  # adjust if hosted elsewhere
pp = pprint.PrettyPrinter(indent=2, width=100)


def unwrap_tool_result(resp):
    """
    Safely unwraps the content from a FastMCP tool call result object.
    """
    if hasattr(resp, "content") and resp.content:
        # The content is a list containing a single content object
        content_object = resp.content[0]

        # It could be JSON or plain text
        if hasattr(content_object, "json"):
            return content_object.json

        if hasattr(content_object, "text"):
            try:
                # Safely evaluate a string containing a Python literal
                return ast.literal_eval(content_object.text)
            except (ValueError, SyntaxError):
                # If it's not a literal, return raw text
                return content_object.text

    return resp


async def main():
    transport = StreamableHttpTransport(url=SERVER_URL)
    client = Client(transport)

    print("\n🚀 Connecting to FastMCP server at:", SERVER_URL)

    async with client:
        # 1. Ping to test connectivity
        print("\n🔗 Testing server connectivity...")
        await client.ping()
        print("✅ Server is reachable!\n")

        # 2. Discover server capabilities
        print("🛠️  Available tools:")
        pp.pprint(await client.list_tools())

        print("\n📚 Available resources:")
        pp.pprint(await client.list_resources())

        print("\n💬 Available prompts:")
        pp.pprint(await client.list_prompts())

        # 3. Fetch the topics resource
        print("\n\n📖 Fetching resource: resource://ai/arxiv_topics")
        res = await client.read_resource("resource://ai/arxiv_topics")
        topics = ast.literal_eval(res[0].text) # type: ignore

        print("Today's AI topics:")
        for i, t in enumerate(topics, 1):
            print(f"  {i}. {t}")

        # # 4. Test the search tool
        # print("\n\n🔍 Testing tool: search_arxiv")
        # raw_search = await client.call_tool(
        #     "search_arxiv",
        #     {"query": "Transformer interpretability", "max_results": 3},
        # )

        # search_results = unwrap_tool_result(raw_search)
        # for i, paper in enumerate(search_results, 1):
        #     print(f"  {i}. {paper['title']}\n     {paper['url']}")

        # 5. Test the summarize tool on the first result
        # if search_results:
        #     first_url = search_results[0]["url"]
        #     print("\n\n📝 Testing tool: summarize_paper")

        #     raw_summary = await client.call_tool(
        #         "summarize_paper",
        #         {"paper_url": first_url},
        #     )

        #     summary = unwrap_tool_result(raw_summary)
        #     print("\nSummary of first paper:\n", summary)

        # 6. Test the prompt generator
        print("\n\n🚀 Testing prompt: explore_topic_prompt")
        prompt_resp = await client.get_prompt(
            "explore_topic_prompt",
            {"topic": "Transformer interpretability"},
        )

        print("\nGenerated prompt for an LLM:")
        for msg in prompt_resp.messages:
            print(f"{msg.role.upper()}: {msg.content.text}\n") # type: ignore


if __name__ == "__main__":
    asyncio.run(main())
