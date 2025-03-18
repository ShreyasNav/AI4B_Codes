import os
os.environ["HF_TOKEN"] = ""
os.environ["HF_DATASETS_CACHE"] = "/projects/data/ttsteam/repos/vllm"
os.environ["HF_HOME"] = "/projects/data/ttsteam/repos/vllm"
os.environ["TRANSFORMERS_CACHE"] = "/projects/data/ttsteam/repos/vllm"
import requests
import json
from googlesearch import search
import trafilatura
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pandas as pd
import time
from bs4 import BeautifulSoup
from yt_dlp import YoutubeDL
import concurrent.futures
import threading
from tqdm import tqdm
from collections import defaultdict

class SearchTerms:
  def __init__(self):

        self.embeddings = HuggingFaceEmbeddings()
        self.data_for_context = []
        self.output_file1name = "data_for_context.json"
        self.search_results = []
        self.output_file2name = "search_terms_with_source.json"
        self.lock = threading.Lock()  


  def find_urls(self, search_queries, serp_api_key, num = 5):

        serp_base_url = "https://serpapi.com/search"
        blacklist_domains = ["www.facebook.com","jiosaavn.com", "spotify.com", "gaana.com", "instagram.com"]

        all_urls_with_queries = []  # Store URLs with their originating queries
        for query in search_queries:
            try:
                params = {
                    "q": query,
                    "num": num,
                    "api_key": serp_api_key
                }
                response = requests.get(serp_base_url, params=params)
                response.raise_for_status()
                search_results = response.json()
                
                if 'organic_results' in search_results:
                    urls = [
                        result.get("link")
                        for result in search_results["organic_results"]
                        if result.get("link")
                    ]

                    # Filter out blacklisted domains and store with query information
                    filtered_urls = [
                        {"url": url, "query": query}
                        for url in urls
                        if not any(blacklisted in url for blacklisted in blacklist_domains)
                    ]

                    all_urls_with_queries.extend(filtered_urls)
                else:

                    print(f"No results found for query: {query}")

                time.sleep(2)
            except Exception as e:
                print(f"Error in SERP API request for query '{query}': {e}")
        yt_links = []
        for each in all_urls_with_queries:
          if "www.youtube.com" in each["url"]:
            yt_links.append({'url': each['url'],
            'query': each['query']})
            all_urls_with_queries.remove(each)

        return all_urls_with_queries, yt_links
    
  def get_yt_video_info(self, yt_links):
      ydl_opts = {
          'quiet': True,
          'extract_flat': True,
      }
      with YoutubeDL(ydl_opts) as ydl:
        for each in yt_links:
            url = each['url']
            info = ydl.extract_info(url, download=False)
            if info:
                each['title'] = info.get('title', 'Unknown')
                each['channel'] = info.get('uploader', 'Unknown')
      with open("yt_direct_data.json", 'w', encoding='utf-8') as f:
          json.dump({
              "search_terms": yt_links
          }, f, indent=2, ensure_ascii=False)

      print("Got direct data for",len(yt_links),"yt links")     

  def extract_text(self, url_info):
    try:
        downloaded = trafilatura.fetch_url(url_info["url"])
        if downloaded:
            text = trafilatura.extract(downloaded,
                                      include_comments=False,
                                      include_tables=False)
            if text:
                cleaned_text = self.clean_text(text)
                if cleaned_text:
                    return {
                        "text": cleaned_text,
                        "url": url_info["url"],
                        "query": url_info["query"]                    }
        return None
    except Exception as e:
        print(f"Error extracting text from {url_info['url']}: {e}")
        return None

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

  def get_context(self, text, search_queries):

        if not text:
            raise ValueError("No documents to process!")

        print(f"\nProcessing {len(text)} documents...")
"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        texts = []
        for doc in text:
            chunks = splitter.split_text(doc["text"])
            filtered_chunks = [
                chunk for chunk in chunks
                if len(chunk.split()) >= 20
                and not any(boilerplate in chunk.lower()
                            for boilerplate in ['cookie', 'privacy', 'terms of service'])
            ]

            texts.extend([{
                "text": chunk,
                "url": doc["url"],
                "query": doc["query"]
            } for chunk in filtered_chunks])

        print(f"\nCreated {len(texts)} text chunks")
"""

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        # List to collect combined chunks with their associated URLs
        combined_chunks = []

        # Combine consecutive small documents until reaching 5000 characters
        temp_text = ""
        temp_urls = set()  # Use a set to track multiple URLs for merged chunks

        for doc in text:
            temp_text += doc["text"] + "\n"  # Append text with separator
            temp_urls.add(doc["url"])        # Track URL

            # If the combined text exceeds the desired chunk size
            if len(temp_text) >= 5000:
                # Split and filter the resulting chunks
                chunks = splitter.split_text(temp_text)
                filtered_chunks = [
                    chunk for chunk in chunks
                    if len(chunk.split()) >= 20
                    and not any(boilerplate in chunk.lower()
                                for boilerplate in ['cookie', 'privacy', 'terms of service'])
                ]

                # Store filtered chunks with corresponding URLs
                combined_chunks.extend([
                    {
                        "text": chunk,
                        "url": ", ".join(temp_urls),  # Combine URLs
                        "query": doc["query"]
                    } for chunk in filtered_chunks
                ])

                # Reset for the next batch of documents
                temp_text = ""
                temp_urls = set()

        # Edge Case: Add remaining text if it never reached 5000 characters
        if temp_text.strip():
            combined_chunks.append({
                "text": temp_text.strip(),
                "url": ", ".join(temp_urls),
                "query": doc["query"]
            })

        print("Total combined chunks:", len(combined_chunks))

"""
        if not texts:
            raise ValueError("No meaningful text chunks found after filtering!")

        vectorstore = FAISS.from_texts(
            [t["text"] for t in texts],
            self.embeddings,
            metadatas=[{"url": t["url"], "query": t["query"]} for t in texts]
        )

        all_contexts = []
        for query in search_queries:
            retrieved_docs = vectorstore.similarity_search(query, k=2)
            all_contexts.extend([doc.page_content for doc in retrieved_docs])

        seen = set()
        unique_contexts = [x for x in all_contexts if not (x in seen or seen.add(x))]

        context = "\n\n".join(unique_contexts)

        for each in text:
          each["context"] = context

        self.data_for_context = text

        print("Context :")
        print(context)
        self.context = context
        return context
"""

  def query_LLM1(self, context, language, num, nvidia_llama_apikey):

        prompt = f"""
        You are an expert in linguistic search optimization. Your task is to generate {num} **highly relevant and specific** YouTube search terms to find high-quality {language} language audio content. 

        Context: 
        {context}

        Guidelines for Generating Search Terms:
        Extract Specific & Unique Terms:  
        - Identify and include **proper nouns, names of people, programs, interviews, and podcasts** directly mentioned in the context.  
        - If no proper nouns are present, generate highly relevant terms based on the linguistic and cultural aspects of {language}.  

        Ensure Linguistic & Cultural Relevance:
        - Incorporate **traditional, modern, and culturally significant** terms used by {language} speakers.  
        - Focus on **speech, interviews, discussions, and storytelling**—avoid generic, broad, or unrelated terms.  

        Strictly Avoid:
        - **Music-related terms**  
        - **Overly generic words** 
        - **Irrelevant or ambiguous terms** that don't improve search specificity.  

        Output Format:  
        - Each term should be **unique** and **enclosed in double quotes**, with **one term per line**.  
        - Do **not** add explanations, numbering, or any extra text—only return the search terms.  

        Now, generate the best {num} search terms based on the given context.

        """


        API_URL = "http://localhost:8090/v1/completions"  # Standard completions endpoint

        data = {
            "model": "meta-llama/Llama-3.3-70B-Instruct",  # Ensure correct name
            "prompt": prompt,
            "max_tokens": 100,
            "temperature":0.7,
            "top_p": 0.8
        }
        response = requests.post(API_URL, json=data)
        try:
            response_text = response.json()['choices'][0]['text']
        except:
            print (response.json())
            breakpoint()
        search_terms = [
            term.strip().strip('"')
            for term in response_text.split('\n')
            if term.strip() and term.strip() != "No search terms found"
        ]

        for each in self.data_for_context:
          each["basic_search_terms"] = search_terms

        with open(self.output_file1name, 'w', encoding='utf-8') as f:
            json.dump({
                "search_terms": self.data_for_context
            }, f, indent=2, ensure_ascii=False)

        return search_terms

  def make_chunks(self, text):
        if not text:
            raise ValueError("No documents to process!")

        print(f"\nProcessing {len(text)} documents...")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        processed_chunks = []
        for doc in text:
            chunks = splitter.split_text(doc["text"])
            filtered_chunks = [
                chunk for chunk in chunks
                if len(chunk.split()) >= 20
                and not any(boilerplate in chunk.lower()
                           for boilerplate in ['cookie', 'privacy', 'terms of service'])
            ]

            processed_chunks.extend([{
                "text": chunk,
                "url": doc["url"],
                "query": doc["query"]
            } for chunk in filtered_chunks])

        print(f"\nCreated {len(processed_chunks)} text chunks")

        if not processed_chunks:
            raise ValueError("No meaningful text chunks found after filtering")

        return processed_chunks

  def process_chunk(self, chunk, language, API_URL, MODEL_NAME, progress_bar):
        """Function to send request and process LLaMA output"""
        prompt = f"""
            Context about {language} language content:
            {chunk['text']}

            Task: Find as many relevant YouTube-specific search terms that would help find {language} language audio content and return me top few (around 5, strictly less than 7) unique and interesting search terms.
            From given text search terms should be related to-
            Names of {language} podcasters
            Names of Speech and speakers in {language}
            Names of people giving and taking interviews in {language}
            Names of famous {language} personalities
            1. Include actual names and proper nouns if mentioned in the text
            2. Use keywords specific to {language} content and name of some person
            3. Include terms that {language} speakers might use, don't include anything related to music and songs in the search terms because these are intended to get high quality {language} audio and speech for training
            4. Don't give generic terms
            Format each term on a new line as:
            "term"
            Do not do numbering of terms and dont include any text other than search terms.
        """

        data = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "max_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.8
        }

        try:
            response = requests.post(API_URL, json=data)
            response.raise_for_status()
            response_text = response.json().get('choices', [{}])[0].get('text', '')

            search_terms = [
                term.strip().strip('"')
                for term in response_text.split("\n")
                if term.strip() and term.strip() != "No search terms found"
            ]

            results = [
                {
                    "search_term": term,
                    "source_chunk": chunk["text"],
                    "source_url2": chunk["url"],
                    "search_query2": chunk["query"]
                }
                for term in search_terms
            ]

            # Update global results safely
            with self.lock:
                self.search_results.extend(results)
                progress_bar.update(1)  # Update progress bar

            return results

        except Exception as e:
            print(f"Error processing chunk: {e}")
            return []

  def query_LLM2(self, chunks, language):
        API_URL = "http://localhost:8090/v1/completions"
        MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"

        with tqdm(total=len(chunks), desc="Processing Chunks", unit="chunk") as progress_bar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                future_to_chunk = {
                    executor.submit(self.process_chunk, chunk, language, API_URL, MODEL_NAME, progress_bar): chunk
                    for chunk in chunks
                }
                for future in concurrent.futures.as_completed(future_to_chunk):
                    future.result()  # Ensure all tasks are completed

        # Save results to a JSON file
        with open(self.output_file2name, 'w', encoding='utf-8') as f:
            json.dump({"search_terms": self.search_results}, f, indent=2, ensure_ascii=False)

        return len(self.search_results)

def main():
  pipeline = SearchTerms()
  language = "Dogri"
  search_queries = [
        f"{language} children story",
        f"{language} bedtime story",
        f"moral stories in {language}",
        f"Stories in {language}  for kids",
        f"{language} fable narration",
        f"kids storytelling in {language}"
    ] #enter the search queries as a list

  serp_api_key = "bd2828baec1881f952b85b33746b20020e4da76e209020d02e293cd3583a6a4f"

  urls, yt_links = pipeline.find_urls(search_queries, serp_api_key, num = 7)
  with open("yt_links.json", 'w', encoding='utf-8') as f:
      json.dump({
          "yt_links": yt_links
      }, f, indent=2, ensure_ascii=False)

  #pipeline.get_yt_video_info(yt_links)

  text = []
  for i, url_info in enumerate(urls, 1):

      result = pipeline.extract_text(url_info)
      if result and len(result["text"]) > 50:
          print(f"✓ Document {i}: Got {len(result['text'])} characters from {result['url']}")
          text.append(result)
      else:
          print(f"✗ Document {i}: Failed or insufficient text from {url_info['url']}")
  print(f"\nSuccessfully extracted text from {len(text)} documents")

  combined_chunks = pipeline.get_context(text, search_queries)

  nvidia_llama_apikey = "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"
  
  basic_search_terms = pipeline.query_LLM1(combined_chunks, language, 35, nvidia_llama_apikey)
  print("Number of basic_search_terms", len(basic_search_terms))
  print(basic_search_terms)
  
  good_queries = search_queries + basic_search_terms 
  good_urls, yt = pipeline.find_urls(good_queries, serp_api_key, num = 15)
 
  good_text = []
  for i, url_info in enumerate(good_urls, 1):
      
      result = pipeline.extract_text(url_info)
      if result and len(result["text"]) > 50:
          print(f"✓ Document {i}: Got {len(result['text'])} characters from {result['url']}")
          good_text.append(result)
      else:
          print(f"✗ Document {i}: Failed or insufficient text from {url_info['url']}")
  print(f"\nSuccessfully extracted text from {len(text)} documents")

  chunks = pipeline.make_chunks(good_text)
  
  final_search_terms = pipeline.query_LLM2(chunks, language)
  print(final_search_terms,"search terms generated")

if __name__ == "__main__":
  main()
