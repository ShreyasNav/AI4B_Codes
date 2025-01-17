import os
import requests
from googlesearch import search
import trafilatura
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pandas as pd
import time
from bs4 import BeautifulSoup
from openai import OpenAI

class SearchTermDiscovery:
    def __init__(self, api_key):
        self.embeddings = HuggingFaceEmbeddings()
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )

    def get_urls_from_google(self, language, num_results=50):
        """Get URLs from Google search using different queries."""
        search_queries = [
            f"{language} stories audio",
            f"{language} audiobooks",
            f"{language} literature",
            f"{language} poets and poems",
            f"{language} folk tales",
            f"{language} novels",
            f"{language} authors"
        ]

        all_urls = set()
        for query in search_queries:
            try:
                urls = search(query, num_results=num_results // len(search_queries), timeout=30)
                all_urls.update(urls)
                time.sleep(2)
            except Exception as e:
                print(f"Error in Google search for query '{query}': {e}")

        return list(all_urls)

    def extract_text_from_url(self, url):
        """Extract text content from a URL using trafilatura."""
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                text = trafilatura.extract(downloaded,
                                           include_comments=False,
                                           include_tables=False)
                if text:
                    cleaned_text = self.clean_text(text)
                    return cleaned_text if cleaned_text else ""
            return ""
        except Exception as e:
            print(f"Error extracting text from {url}: {e}")
            return ""

    def clean_text(self, text):
        """Clean extracted text with improved filtering."""
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            if any([
                len(line) < 20,
                'download' in line.lower(),
                'cookie' in line.lower(),
                'privacy policy' in line.lower(),
                'terms of use' in line.lower(),
                'all rights reserved' in line.lower(),
                line.startswith('©'),
                line.startswith('http'),
                not any(c.isalpha() for c in line)
            ]):
                continue

            cleaned_lines.append(line)

        return ' '.join(line for line in cleaned_lines if line)

    def gather_context(self, language, num_results=30):
        """Gather context from Google search results."""
        print(f"Gathering URLs for {language}...")
        urls = self.get_urls_from_google(language, num_results)

        print(f"\nFound {len(urls)} URLs. Starting text extraction...")
        documents = []
        for i, url in enumerate(urls, 1):
            text = self.extract_text_from_url(url)
            if text and len(text) > 100:
                print(f"✓ Document {i}: Got {len(text)} characters from {url}")
                documents.append({
                    "text": text,
                    "source": url
                })
            else:
                print(f"✗ Document {i}: Failed or insufficient text from {url}")

        print(f"\nSuccessfully extracted text from {len(documents)} documents")
        return documents

    def process_documents(self, documents):
        """Process documents with improved chunking."""
        if not documents:
            raise ValueError("No documents to process!")

        print(f"\nProcessing {len(documents)} documents...")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        texts = []
        for doc in documents:
            chunks = splitter.split_text(doc["text"])
            filtered_chunks = [
                chunk for chunk in chunks
                if len(chunk.split()) >= 20
                and not any(boilerplate in chunk.lower()
                            for boilerplate in ['cookie', 'privacy', 'terms of service'])
            ]

            texts.extend([{
                "text": chunk,
                "source": doc["source"]
            } for chunk in filtered_chunks])

        print(f"\nCreated {len(texts)} text chunks")

        if not texts:
            raise ValueError("No meaningful text chunks found after filtering!")

        vectorstore = FAISS.from_texts(
            [t["text"] for t in texts],
            self.embeddings,
            metadatas=[{"source": t["source"]} for t in texts]
        )

        return vectorstore

    def query_llm(self, context: str, language: str) -> str:
        """Query NVIDIA-hosted LLaMA LLM with the given context."""
        messages = [
            {"role": "user", "content": f"""
            Available Context about {language} language content:
            {context}

            Task: Find as many relevant YouTube-specific search terms that would help find {language} language audio content and return me 20 unique and interesting search terms.
            Be sure to:
            1. Include actual names of books, programs, or audio series if mentioned in the context
            2. Use keywords specific to {language} content and name of some person
            3. Include traditional and cultural terms that speakers might use
            4. Consider both modern and traditional content types

            Format each term on a new line as:
            "term"
            """
            }
        ]

        completion = self.client.chat.completions.create(
            model="meta/llama-3.3-70b-instruct",
            messages=messages,
            temperature=0.2,
            top_p=0.7,
            max_tokens=1024,
            stream=False
        )

        result = "".join(
            chunk.choices[0].delta.content
            for chunk in completion
            if chunk.choices[0].delta.content is not None
        )

        return result

    def generate_search_terms(self, language, vectorstore):
        """Generate YouTube search terms using improved context retrieval and NVIDIA-hosted LLaMA."""
        print("\nGenerating search terms...")

        search_queries = [
            f"{language} stories audio",
            f"{language} audiobooks",
            f"{language} literature",
            f"{language} poets and poems",
            f"{language} folk tales",
            f"{language} novels",
            f"{language} authors"
        ]

        all_contexts = []
        for query in search_queries:
            retrieved_docs = vectorstore.similarity_search(query, k=4)
            all_contexts.extend([doc.page_content for doc in retrieved_docs])

        seen = set()
        unique_contexts = [x for x in all_contexts if not (x in seen or seen.add(x))]

        context = "\n\n".join(unique_contexts)
        print("context is...")
        print(context)

        result = self.query_llm(context, language)
        print("result is...")
        print(result)

        terms = [
            term.strip() for term in result.split('\n')
            if term.strip() and language.lower() in term.lower()
        ]

        print(f"\nGenerated {len(terms)} search terms")
        return terms

    def validate_terms(self, search_terms):
        """Validate search terms with improved error handling."""
        results = []
        for term in search_terms:
            try:
                url = f"https://www.youtube.com/results?search_query={term}"
                response = requests.get(url)
                soup = BeautifulSoup(response.text, 'html.parser')
                video_count = len(soup.find_all('div', {'class': 'style-scope ytd-video-renderer'}))

                results.append({
                    'search_term': term,
                    'result_count': video_count,
                    'url': url
                })
                time.sleep(2)

            except Exception as e:
                print(f"Error validating term '{term}': {e}")
                continue

        return pd.DataFrame(results).sort_values('result_count', ascending=False)

def main():
    # Initialize with your NVIDIA API key
    pipeline = SearchTermDiscovery(api_key="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")

    language = "Punjabi"
    print(f"\nStarting search term discovery for {language}...")

    documents = pipeline.gather_context(language)
    if not documents:
        print("Error: No documents found!")
        return

    vectorstore = pipeline.process_documents(documents)

    # Generate and validate search terms
    search_terms = pipeline.generate_search_terms(language, vectorstore)
    if not search_terms:
        print("Error: No search terms generated!")
        return

    print("\nGenerated terms:", search_terms)

    ranked_terms = pipeline.validate_terms(search_terms)

if __name__ == "__main__":
    main()
