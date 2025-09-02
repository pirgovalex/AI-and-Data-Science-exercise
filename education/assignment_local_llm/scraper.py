import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

OUTPUT = "scraped_content.txt"
SCRAPED = "links_scraped.txt"

visited = set()

def write_to_file(filename, data):
    with open(filename, 'w') as f:
        f.writelines(url + '\n' for url in data)

def get_links(url, depth, max_depth):
    if depth > max_depth or url in visited:
        return []
    visited.add(url)

    links = []
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raises an exception for bad status code
        blacklist = [
    "copyright", "©", "trademark", "®", "™",
    "terms of use", "privacy policy", "contact us",
    "follow us", "sitemap", "cookie", "all rights reserved",
    "kitware.com", "about", "features", "history", "news & updates",
    "solutions", "support", "training", "getting started",
    "documentation", "customize", "search", "download"
]



        
        soup = BeautifulSoup(response.content, 'lxml')
        texts = [
    tag.get_text() for tag in soup.find_all(['p', 'h1', 'h2', 'h3','ol','ul'])
    if not any(word.lower() in tag.get_text().lower() for word in blacklist)
]
        page_content = "\n".join(texts)



        with open(OUTPUT, "a", encoding="utf-8") as f:
            f.write(page_content + "\n")


        for link in soup.find_all('a', href=True):
            full_url = urljoin(url, link['href'])
            if full_url.startswith(URL): links.append(full_url)
    except requests.RequestException as e:
        print(f"Error scraping {url}: {e}")
        return []

    for link in links:
        get_links(link, depth + 1, max_depth)

    return links

def clean_white_spaces(input_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    with open(input_file, 'w') as f:
        for line in lines:
            if line.strip() != "":
                f.write(line)
def filter_noise(input_text):
    pass
URL = "https://cmake.org"
# def main():
#     get_links(URL, 1, 3)
#     write_to_file(SCRAPED, visited)
#     clean_white_spaces(OUTPUT)

# main()
