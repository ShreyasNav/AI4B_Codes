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
from aksharamukha import transliterate

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

        if isinstance(search_queries[0], str):
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
                            {"url1": url, "query": query}
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
                if "www.youtube.com" in each["url1"]:
                    yt_links.append({'url': each['url1'],
                    'query': each['query']})
                    all_urls_with_queries.remove(each)

            return all_urls_with_queries, yt_links

        elif isinstance(search_queries[0], dict):
            all_url_tracked = []
            for query in search_queries:
                try:
                    params = {
                        "q": query["basic_search_term"],
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

                        for url in urls:
                            if not any(blacklisted in url for blacklisted in blacklist_domains):
                                element = [{"chunk1":query['chunk1'],
                                            "url1":query['url1'],
                                            "query":query['query'],
                                            "basic_search_term":query['basic_search_term'],
                                            "url2":url}]
                        all_url_tracked.extend(element)
                    else:
                        print(f"No results found for query: {query['basic_search_term']}")

                    time.sleep(2)
                except Exception as e:
                    print(f"Error in SERP API request for query '{query}': {e}")
            yt_links = []
            for each in all_url_tracked:
                if "www.youtube.com" in each["url2"]:
                    yt_links.append({'url': each['url2'],
                    'query': each["basic_search_term"]})
                    all_url_tracked.remove(each)

            return all_url_tracked, yt_links
     
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
    if "url2" in url_info:
        try:
            downloaded = trafilatura.fetch_url(url_info["url2"])
            if downloaded:
                text = trafilatura.extract(downloaded,
                                        include_comments=False,
                                        include_tables=False)
                if text:
                    cleaned_text = self.clean_text(text)
                    if cleaned_text:
                        url_info["text"] = cleaned_text
                        return url_info
            return None
        except Exception as e:
            print(f"Error extracting text from {url_info['url2']}: {e}")
            return None
    
    else:
        try:
            downloaded = trafilatura.fetch_url(url_info["url1"])
            if downloaded:
                text = trafilatura.extract(downloaded,
                                        include_comments=False,
                                        include_tables=False)
                if text:
                    cleaned_text = self.clean_text(text)
                    if cleaned_text:
                        url_info["text"] = cleaned_text
                        return url_info
            return None
        except Exception as e:
            print(f"Error extracting text from {url_info['url1']}: {e}")
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

  def process_chunk1(self, each, language, num):
        """Processes a single chunk for search term generation."""
        prompt = f"""
        You are an expert in linguistic search optimization. Your task is to generate {num} **highly relevant and specific** YouTube search terms to find high-quality {language} language audio content. 

        Context: 
        {each['chunk1']}

        Follow given instructions striclty
        Guidelines for Generating Search Terms:
        Extract Specific & Unique Terms:  
        - Identify and include **proper nouns, names of people and programs** directly mentioned in the context.  
        - If no proper nouns are present, generate highly specific search terms that may be derived from the context  

        Ensure Linguistic & Cultural Relevance:
        - Incorporate **traditional, modern, and culturally significant** terms used by {language} speakers.  

        Strictly Avoid:
        - **Music and songs retaled terms**  
        - **Generic terms** 
        - **Irrelevant or ambiguous terms** that don't improve search specificity.  

        Output Format:  
        - Each term should be **unique** and **enclosed in double quotes**, with **one term per line**.  
        -Format each term on a new line as:
        "term"
        Do not do numbering of terms and dont include any text other than search terms.
        Give this output ina csv file

        Now, strictly generate the best search terms based on the given context.
        """

        API_URL = "http://localhost:8090/v1/completions"

        data = {
            "model": "meta-llama/Llama-3.3-70B-Instruct",
            "prompt": prompt,
            "max_tokens": 100,
            "temperature": 0.6,
            "top_p": 0.8
        }

        try:
            response = requests.post(API_URL, json=data)
            response_text = response.json().get('choices', [{}])[0].get('text', '')

            search_terms = [
                term.strip().strip('"')
                for term in response_text.split('\n')
                if term.strip() and term.strip() != "No search terms found"
            ]

            return [
                {
                    "query": each['query'], 
                    "url1": each['url1'], 
                    "chunk1": each['chunk1'], 
                    "basic_search_term": i
                }
                for i in search_terms
            ]

        except Exception as e:
            print(f"Error processing chunk: {e}")
            return []

  def query_LLM1(self, chunks, language, num):

        basic_search_terms = []  # Local variable for results

        with tqdm(total=len(chunks), desc="Processing Chunks", unit="chunk") as progress_bar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                future_to_chunk = {
                    executor.submit(self.process_chunk1, each, language, num): each
                    for each in chunks
                }

                for future in concurrent.futures.as_completed(future_to_chunk):
                    result = future.result()  # Collect each chunk's result
                    with self.lock:          # Ensure thread-safe accumulation
                        basic_search_terms.extend(result)  
                    progress_bar.update(1)   # Update progress bar

        return basic_search_terms

  def transliterate(search_terms, target_lang):
    transliterated = copy.(search_terms)
    for term in transliterated:
        term[transliterated] = transliterate.process('IAST', target_lang, term) 

    return transliterated

  def translate(search_terms, source_lang, target_lang):
    translated = copy.(search_terms)
    transliterator = Transliterator(source_lang=source_lang, target_lang=target_lang)
    for term in translated:
        term[translated] = transliterator.transliterate(term)
    
    return translated

  def make_chunks(self, text):
        if "url2" not in text[0]:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=5000,
                chunk_overlap=100,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )

            combined_chunks = []

            temp_text = ""
            temp_urls = set()  

            for doc in text:
                temp_text += doc["text"] + "\n"  
                temp_urls.add(doc["url1"])       

                if len(temp_text) >= 5000:
                    chunks = splitter.split_text(temp_text)
                    filtered_chunks = [
                        chunk for chunk in chunks
                        if len(chunk.split()) >= 20
                        and not any(boilerplate in chunk.lower()
                                    for boilerplate in ['cookie', 'privacy', 'terms of service'])
                    ]

                    combined_chunks.extend([
                        {
                            "chunk1": chunk,
                            "url1": ", ".join(temp_urls),  
                            "query": doc["query"]
                        } for chunk in filtered_chunks
                    ])

                    temp_text = ""
                    temp_urls = set()
            # Edge Case
            if temp_text.strip():
                combined_chunks.append({
                    "chunk1": temp_text.strip(),
                    "url1": ", ".join(temp_urls),
                    "query": doc["query"]
                })

            print("Total chunks:", len(combined_chunks))

            return combined_chunks 
        
        if "url2" in text[0]:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=5000,
                chunk_overlap=100,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )

            combined_chunks = []

            temp_text = ""
            temp_urls = set() 

            for doc in text:
                temp_text += doc["text"] + "\n"  
                temp_urls.add(doc["url2"])     

                if len(temp_text) >= 5000:
                    chunks = splitter.split_text(temp_text)
                    filtered_chunks = [
                        chunk for chunk in chunks
                        if len(chunk.split()) >= 20
                        and not any(boilerplate in chunk.lower()
                                    for boilerplate in ['cookie', 'privacy', 'terms of service'])
                    ]

                    combined_chunks.extend([
                        {   
                            "chunk2":chunk,
                            "url2": ", ".join(temp_urls), 
                            "query": doc["query"],
                            "url1":doc['url1'],
                            "chunk1":doc['chunk1'],
                            "basic_search_term":doc["basic_search_term"]
                        } for chunk in filtered_chunks
                    ])

                    temp_text = ""
                    temp_urls = set()

            # Edge Case
            if temp_text.strip():
                combined_chunks.append({
                    "chunk2": temp_text.strip(),
                    "url2": ", ".join(temp_urls),
                    "query": doc["query"],
                    "url1": doc.get('url1', ''),
                    "chunk1": doc.get('chunk1', ''),
                    "basic_search_term": doc.get("basic_search_term", '')
                })

            print("Total combined chunks:", len(combined_chunks))

            return combined_chunks 

  def process_chunk(self, chunk, language, num, API_URL, MODEL_NAME, progress_bar):

        prompt = f"""
            Context about {language} language content:
            {chunk['chunk2']}

            Follow given instructions striclty
            Task: Find as many relevant YouTube-specific search terms that would help find {language} language audio content and return me striclty top {num} unique and interesting search terms.
            From given text search terms should -
            1. Include actual names of people, names of programs and proper nouns if mentioned in the text
            2. Use keywords specific to {language} content and name of some person
            3. Include terms that {language} speakers might use, don't include anything related to music and songs in the search terms because these are intended to get high quality {language} audio and speech for training TTS model
            4. Don't give generic terms
            Format each term on a new line as:
            "term"
            Do not do numbering of terms and dont include any text other than search terms.
            Give this output in a csv file
        """

        data = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "max_tokens": 100,
            "temperature": 0.6,
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
                    "chunk2": chunk['chunk2'],
                    "url2": chunk['url2'],
                    "query": chunk["query"],
                    "url1":chunk['url1'],
                    "chunk1":chunk['chunk1'],
                    "basic_search_term":chunk["basic_search_term"],
                    "search_term": term
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

  def query_LLM2(self, chunks, language, num):
        API_URL = "http://localhost:8090/v1/completions"
        MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"

        with tqdm(total=len(chunks), desc="Processing Chunks", unit="chunk") as progress_bar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                future_to_chunk = {
                    executor.submit(self.process_chunk, chunk, language, num, API_URL, MODEL_NAME, progress_bar): chunk
                    for chunk in chunks
                }
                for future in concurrent.futures.as_completed(future_to_chunk):
                    future.result()  # Ensure all tasks are completed

        # Save results to a JSON file
        with open(self.output_file2name, 'w', encoding='utf-8') as f:
            json.dump({"search_terms": self.search_results}, f, indent=2, ensure_ascii=False)

        return len(self.search_results)

  def filter_redundant_terms(search_terms, threshold=0.5):

        filtered_terms = []
        for term in search_terms:
            words1 = term.lower().split()
            for exist_term in filtered_terms:
                words2 = exist_term.lower().split()
                count = 0
                for i in words1:
                    for j in words2:
                        if  Levenshtein.ratio(i, j) > threshold:
                            count+=1
            if count < 2:
                filtered_terms.append(term)

        return filtered_terms

def main():
  pipeline = SearchTerms()
  language = "Dogri"
  search_queries = [
        f"Children stories and books in {language}",
    ] #enter the search queries as a list

  serp_api_key = "155436a866a67e84ca58ca77806f38348a0678ef5b1494175a6843865b87d4ce"

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
          print(f"✓ Document {i}: Got {len(result['text'])} characters from {result['url1']}")
          text.append(result)
      else:
          print(f"✗ Document {i}: Failed or insufficient text from {url_info['url1']}")
  print(f"\nSuccessfully extracted text from {len(text)} documents")

  chunks = pipeline.make_chunks(text)
  
  basic_search_terms = pipeline.query_LLM1(chunks, language, 7)
  print("Number of basic_search_terms", len(basic_search_terms))
  with open("basic_search_terms.txt", "w") as file:
    for each in basic_search_terms:
        file.write(f"{each['basic_search_term']}, /n")

#   transliterated_terms = pipeline.transliterate(basic_search_terms, target_lang='Gurmukhi') 
#   translated_terms = pipeline.translate(transliterated_terms, source_lang='eng', target_lang='pan')

#   basic_searchterms_processed = basic_search_terms + transliterated_terms + translated_terms
  #good_queries = search_queries + basic_search_terms 
  good_urls, yt = pipeline.find_urls(basic_search_terms, serp_api_key, num = 15)
 
  good_text = []
  for i, url_info in enumerate(good_urls, 1):
      
      result = pipeline.extract_text(url_info)
      if result and len(result["text"]) > 50:
          print(f"✓ Document {i}: Got {len(result['text'])} characters from {result['url2']}")
          good_text.append(result)
      else:
          print(f"✗ Document {i}: Failed or insufficient text from {url_info['url2']}")
  print(f"\nSuccessfully extracted text from {len(good_text)} documents")

  chunks = pipeline.make_chunks(good_text)
  
  final_search_terms = pipeline.query_LLM2(chunks, language, num = 15)
  print(final_search_terms,"search terms generated")

if __name__ == "__main__":
  main()
