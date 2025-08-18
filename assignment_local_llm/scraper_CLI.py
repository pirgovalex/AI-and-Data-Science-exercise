import argparse
from scraper import get_links, write_to_file, clean_white_spaces, URL, OUTPUT, SCRAPED, visited

def main():
    parser = argparse.ArgumentParser(description="Run your scraper with custom URL and depth")
    parser.add_argument("--url", type=str, required=True, help="Starting URL to scrape")
    parser.add_argument("--depth", type=int, default=2, help="Maximum crawl depth (default: 2)")
    args = parser.parse_args()

    global URL
    URL = args.url

    print(f"[info] Starting scrape of {URL} to depth {args.depth}")
    get_links(URL, 1, args.depth)
    write_to_file(SCRAPED, visited)
    clean_white_spaces(OUTPUT)
    print(f"[done] Content saved to {OUTPUT}, visited links saved to {SCRAPED}")

if __name__ == "__main__":
    main()