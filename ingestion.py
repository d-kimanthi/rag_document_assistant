import asyncio
import os
import ssl
from typing import Any, Dict, List
from urllib import response

import certifi
from dotenv import load_dotenv

# from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

from logger import Colors, log_error, log_header, log_info, log_success, log_warning

load_dotenv()

# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    show_progress_bar=False,
    chunk_size=50,
    retry_min_seconds=10,
)

vector_store = PineconeVectorStore(
    index_name="rag-document-assistant",
    embedding=embeddings,
)

tavily_crawl = TavilyCrawl()
tavily_extract = TavilyExtract()
tavily_map = TavilyMap()


async def index_documents(documents: List[Document], batch_size: int) -> None:
    """Index documents in batches"""
    # Create batches
    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    # Process all batches concurrently
    async def add_batch(batch: List[Document], batch_num: int):
        try:
            await vector_store.aadd_documents(batch)
            log_success(
                f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch)} documents)"
            )
        except Exception as e:
            log_error(f"VectorStore Indexing: Failed to add batch {batch_num} - {e}")
            return False
        return True

    # Process batches concurrently
    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successful batches
    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log_success(
            f"VectorStore Indexing: All batches processed successfully! ({successful}/{len(batches)})"
        )
    else:
        log_warning(
            f"VectorStore Indexing: Processed {successful}/{len(batches)} batches successfully"
        )


async def main():
    """Main function"""
    log_header("Ingesting documents")
    tavily_crawl_results = tavily_crawl.invoke(
        {
            "url": "https://python.langchain.com",
            "max_depth": 3,
            "extract_depth": "advanced",
            "instructions": "Documentatin relevant to ai agents",
        }
    )
    all_documents = [
        Document(page_content=doc["raw_content"], metadata={"source": doc["url"]})
        for doc in tavily_crawl_results["results"]
    ]
    log_success(f"Crawled {len(all_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    split_documents = text_splitter.split_documents(all_documents)
    log_success(f"Split into {len(split_documents)} chunks")

    await index_documents(split_documents, batch_size=500)

    log_header("PIPELINE COMPLETE")
    log_success("🎉 Documentation ingestion pipeline finished successfully!")
    log_info("📊 Summary:", Colors.BOLD)
    log_info(f"   • Pages crawled: {len(tavily_crawl_results)}")
    log_info(f"   • Documents extracted: {len(all_documents)}")
    log_info(f"   • Chunks created: {len(split_documents)}")


if __name__ == "__main__":
    asyncio.run(main())
