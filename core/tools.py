from langchain_core.tools import tool
from typing import Optional, ClassVar
from langchain_core.tools import BaseTool
import traceback
import time
import concurrent.futures
from langchain_exa import ExaSearchRetriever, TextContentsOptions
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from .utils import Utility
import re


class ResilientSearchTool(BaseTool):
    name: ClassVar[str] = "resilient_search"
    description: ClassVar[str] = (
        "Searches the internet using a fallback chain of DuckDuckGo, Tavily, Serper, and Exa."
    )

    duckduckgo: Optional[object] = None
    tavily: Optional[object] = None
    serp: Optional[object] = None
    exa: Optional[object] = None
    timeout: int = 5  # max per-tool wait time

    def __init__(self, duckduckgo=None, tavily=None, serp=None, exa=None, timeout=5):
        super().__init__()
        self.duckduckgo = duckduckgo
        self.tavily = tavily
        self.serp = serp
        self.exa = self.exa_search()
        self.timeout = timeout

    def exa_search(self):
        llm = Utility.llm()
        retriever = ExaSearchRetriever(
            k=3, text_contents_options=TextContentsOptions(max_characters=200)
        )
        prompt = PromptTemplate.from_template(
            """You are the trained assistant which see the content and elaborate using you own knowledge also about the query. This is the following context and the respective query:
            query: {query}
            <context>
            {context}
            </context"""
        )
        chain = (
            RunnableParallel({"context": retriever, "query": RunnablePassthrough()})
            | prompt
            | llm
        )
        return chain

    def _attempt_with_timeout(self, func, query: str) -> str:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, query)
            return future.result(timeout=self.timeout)

    def _run(self, query: str, run_manager=None) -> str:
        tools = []

        if self.duckduckgo:
            tools.append(("DuckDuckGo", lambda q: self.duckduckgo.invoke(q)))
        if self.serp:
            tools.append(("Serper", lambda q: self.serp.run(q)))
        if self.exa:
            tools.append(("Exa", lambda q: self.invoke(f"{q}?").content))
        if self.tavily:
            tools.append(
                (
                    "Tavily",
                    lambda q: self.tavily.invoke({"query": q})["results"][0]["content"],
                )
            )

        for name, func in tools:
            try:
                print(f"🔍 Trying {name}...")
                start = time.time()
                result = self._attempt_with_timeout(func, query)
                elapsed = round(time.time() - start, 2)
                print(f"✅ {name} succeeded in {elapsed}s.")
                return f"{result}"
            except Exception as e:
                print(f"❌ {name} failed: {e}")
                print(traceback.format_exc())

        return "Failed to Access The Information, Please Try Again"
